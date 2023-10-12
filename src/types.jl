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

Implementation must be passed by type because they are all abstract types to
allow for hierarchy.

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

Converting a bounded set `Ω` to other type parameters `T` and/or `N` can be
done by `convert` or by:

    BoundedSet{T}(Ω)
    BoundedSet{T,N}(Ω)

See [`NumOptBase.Bound`](@ref) for possible bound arguments.

"""
struct BoundedSet{T,N,L<:Bound{T,N},U<:Bound{T,N}}
    lower::L
    upper::U
end

function BoundedSet{T,N}(lower::L, upper::U) where {T,N,L<:Bound{T,N},U<:Bound{T,N}}
    # FIXME: Check whether set is feasible.
    return BoundedSet{T,N,L,U}(lower, upper)
end
function BoundedSet{T,N}(lower, upper) where {T,N}
    to_bound(::Type{T}, ::Val{N}, B::Bound{T,N}) where {T,N} = B
    to_bound(::Type{T}, ::Val{N}, B::Number) where {T,N} = as(T, B)
    to_bound(::Type{T}, ::Val{N}, B::AbstractArray{<:Any,N}) where {T,N} =
        copyto!(similar(B, T), B)
    @noinline to_bound(::Type{T}, ::Val{N}, B) where {T,N} =
        throw(ArgumentError(
            "cannot convert argument of type $(typeof(B)) into a bound of type Bound{$T,$N}"))
    return BoundedSet{T,N}(to_bound(T, Val(N), lower),
                           to_bound(T, Val(N), upper))
end

BoundedSet{T}(Ω::BoundedSet{T,N}) where {T,N} = Ω
BoundedSet{T}(Ω::BoundedSet{<:Any,N}) where {T,N} = BoundedSet{T,N}(Ω)

BoundedSet{T,N}(Ω::BoundedSet{T,N}) where {T,N} = Ω
BoundedSet{T,N}(Ω::BoundedSet) where {T,N} = BoundedSet{T,N}(Ω.lower, Ω.upper)

Base.convert(::Type{T}, Ω::T) where {T<:BoundedSet} = Ω
Base.convert(::Type{BoundedSet{T}}, Ω::BoundedSet) where {T} = BoundedSet{T}(Ω)
Base.convert(::Type{BoundedSet{T,N}}, Ω::BoundedSet) where {T,N} = BoundedSet{T,N}(Ω)
