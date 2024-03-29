"""
    NumOptBase.RealComplex{R<:Real}

is the type union of `R` and `Complex{R}`.

"""
const RealComplex{R<:Real} = Union{R,Complex{R}}

"""
    NumOptBase.Identity()

yields a singleton object representing the identity mapping for the
[`NumOptBase.apply!`](@ref) method.

"""
struct Identity end

"""
    NumOptBase.Id

is the singleton object representing the identity mapping for the
[`NumOptBase.apply!`](@ref) method.

"""
const Id = Identity()

"""
    NumOptBase.Diag(A)

yields an object behaving as a diagonal linear mapping for the
[`NumOptBase.apply!`](@ref) method.

"""
struct Diag{T,N,A<:AbstractArray{T,N}}
    diag::A
    Diag{T,N,A}(x::A) where {T,N,A<:AbstractArray{T,N}} = new{T,N,A}(x)
end
LinearAlgebra.diag(A::Diag) = A.diag

# Other constructors.
Diag(x::A) where {T,N,A<:AbstractArray{T,N}} = Diag{T,N,A}(x)
Diag{T}(x::AbstractArray{T}) where {T} = Diag(x)
Diag{T}(x::AbstractArray) where {T} = Diag(convert(AbstractArray{T}, x))
Diag{T,N}(x::AbstractArray{T,N}) where {T,N} = Diag(x)
Diag{T,N}(x::AbstractArray) where {T,N} = Diag(convert(AbstractArray{T,N}, x))
Diag{T,N,A}(x::AbstractArray) where {T,N,A<:AbstractArray{T,N}} = Diag(convert(A, x))

Base.convert(::Type{T}, x::T) where {T<:Diag} = x
Base.convert(::Type{T}, x) where {T<:Diag} = T(diag(x))

"""
    NumOptBase.Engine

is the abstract type inherited by numerical *engines* used for computations.
Numerical engines allows different implementations to co-exist in a Julia
session. The caller of a `NumOptBase` methods may choose a specific engine,
otherwise, a suitable engine is chosen based on the type of the arguments.

Implementation must be passed by type because engine types are all abstract
types to allow for hierarchy.

Explicit loops:

* `LoopEngine` - simple Julia loop with bound checking;

* `InBoundsLoopEngine` - Julia loop without bound checking (`@inbounds`);

* `SimdLoopEngine` - Julia loop without bound checking and with SIMD (Single
  Instruction Multiple Data) vectorization (`@inbounds @simd`);

* `TurboLoopEngine` - Julia loop without bound checking and with AVX
  vectorization (`@avx` or `@turbo`).

GPU arrays:

* `CudaEngine` - implementation suitable for `CuArray`.

Fall-back:

* `Engine` - super-type of all engine types.

"""
abstract type Engine end

"""
    NumOptBase.LoopEngine <: NumOptBase.Engine

is the abstract type identifying implementation with simple loops and bound
checking. Fallback to [`NumOptBase.Engine`](@ref) if no implementation for that
engine exists.

"""
abstract type LoopEngine <: Engine end

"""
    NumOptBase.InBoundsLoopEngine <: NumOptBase.LoopEngine

is the abstract type identifying implementation with simple in-bounds loops
(i.e. `@inbounds`). Fallback to [`NumOptBase.LoopEngine`](@ref) if no
implementation for that engine exists.

"""
abstract type InBoundsLoopEngine <: LoopEngine end

"""
    NumOptBase.SimdLoopEngine <: NumOptBase.InBoundsLoopEngine

is the abstract type identifying implementations with `@simd` loops, that is
using SIMD (Single Instruction Multiple Data) instructions. Fallback to
[`NumOptBase.InBoundsLoopEngine`](@ref) if no implementation for that engine
exists.

"""
abstract type SimdLoopEngine <: InBoundsLoopEngine end

"""
    NumOptBase.SimdArray{T,N}

is the type(s) of arrays suitable for `@simd` optimized loops.

"""
const SimdArray{T,N} = AbstractArray{T,N}

"""
    NumOptBase.SimdLoopEngine <: NumOptBase.InBoundsLoopEngine

is the abstract type identifying implementation with `@avx` or `@turbo` loops.
Fallback to [`NumOptBase.SimdLoopEngine`](@ref) if no implementation for that
engine exists.

"""
abstract type TurboLoopEngine <: SimdLoopEngine end

"""
    NumOptBase.TurboArray{T,N}

is the type(s) of arrays suitable for `@turbo` optimized loops.

"""
const TurboArray{T,N} = StridedArray{T,N}

"""
    NumOptBase.CudaEngine <: NumOptBase.Engine

is the abstract type identifying implementation for CUDA arrays.

"""
abstract type CudaEngine <: Engine end

# The search direction may be negated, this is indicated by argument `pm` which
# is either a `+` or a `-`.
const Plus = typeof(+)
const Minus = typeof(-)
const PlusOrMinus = Union{Plus,Minus}

"""
    Bound{T,N}

is the type of argument suitable to represent a bound for `N`-dimensional
variables of element type `T` in `NumOptBase` package.

Bounds may be specified as:
- `nothing` if the bound is unlimited;
- a scalar if the bound is the same for all variables;
- an array with the same axes as the variables.

Owing to the complexity of managing all possibilities in the methods
implementing bound constraints, bound specified as arrays *conformable* with
the variables are not directly supported. The caller may extend the array of
bound values to the same size as the variables. This may be done in a higher
level interface.

For simplicity and type-stability, only `nothing` is considered as unlimited
bounds even though all lower (resp. upper) bound values may be `-∞` (resp.
`+∞`).

"""
const Bound{T,N} = Union{Nothing,T,AbstractArray{T,N}}

"""
    BoundedSet{T,N}(lower, upper) -> Ω

yields an object `Ω` representing the set of variables bounded below by `lower`
and bounded above by `upper`. Type parameter `T` and `N` are the floating-point
type for computations and the number of dimensions of the variables.

Call `isempty(Ω)` to check whether `Ω` is not feasible. A true result is
returned if `lower .≤ upper` does not hold evrywhere, in particular any `NaN`s
in the bound values will result in an infeasible set.

Call `x ∈ Ω` to check whether `x` belongs to `Ω`.

Converting a bounded set `Ω` to other type parameters `T` and/or `N` can be
done by `convert` or by:

    BoundedSet{T}(Ω)
    BoundedSet{T,N}(Ω)

See [`NumOptBase.Bound`](@ref) for possible bound arguments.

"""
struct BoundedSet{T,N,L<:Bound{T,N},U<:Bound{T,N}}
    lower::L
    upper::U
    function BoundedSet{T,N,L,U}(lower::L, upper::U) where {T,N,L<:Bound{T,N},U<:Bound{T,N}}
        # FIXME: Check whether set is feasible.
        return new{T,N,L,U}(lower, upper)
    end
end

"""
    P = Projector(Ω)

yields a callable object implementing the projection onto subset `Ω`. The
result can be used as:

    xₚ = P(x)

to yield the projection `xₚ` of `x` onto `Ω` or as:

   P(xₚ, x)

to overwrite `xₚ` with the projection of `x` onto `Ω`.

Note that this assumes that [`NumOptBase.project_variables`](@ref) is
applicable for the variables `x` and for objects like `Ω`.

"""
struct Projector{T}
    Ω::T
end
