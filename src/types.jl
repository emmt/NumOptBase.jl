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
const TurboArray{T,N} = Union{StridedArray{T,N},AbstractUniformArray{T,N}}

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
    BoundedSet(lower, upper) -> Ω
    BoundedSet{T}(lower, upper) -> Ω

build an object `Ω` representing the set of variables bounded below by `lower`
and bounded above by `upper`. Type parameter `T` is the element type of the
variables; if omitted, it is inferred from the arguments. Arguments must be
arrays with the same axes. Uniform arrays from the
[`StructuredArray`](https://github.com/emmt/StructuredArrays.jl) package may be
used to represent uniform or unlimiting bounds. For example:

    lower = UniformArray(typemin(T), dims)   # no lower bound
    lower = UniformArray(val, dims)          # all lower bounds equal to value `val`
    upper = UniformArray(typemax(T), dims)   # no upper bound
    upper = UniformArray(val, dims)          # all upper bounds equal to value `val`

There is another constructor to facilitate the building of a bounded set:

    BoundedSet{T}(vars::AbstractArray; lower=nothing, upper=nothing) -> Ω
    BoundedSet(vars::AbstractArray; lower=nothing, upper=nothing) -> Ω

where `vars` is an array of the same element type, size, and axes as the
variables to which the bounds apply, while keywords `lower` and `upper`
respectively specify the lower and upper bounds, as an array, as a scalar value,
or as `nothing` (the default) if the set is not bounded below or above. If type
parameter `T` is omitted, `T = float(eltype(vars))` is assumed.

Call `isempty(Ω)` to check whether `Ω` is not feasible. A true result is
returned if `lower ≤ upper` does not hold element-wise, in particular any
`NaN`s in the bound values will result in an infeasible set.

Call `x ∈ Ω` to check whether `x` belongs to `Ω`.

A bounded set `Ω` is iterable:

    lower, upper = Ω

Converting a bounded set `Ω` to another type parameter `T` can be done by
either of:

    BoundedSet{T}(Ω)
    convert(BoundedSet{T}, Ω)

See [`NumOptBase.Bound`](@ref) for possible bound arguments.

"""
struct BoundedSet{T,N,L<:AbstractArray{T,N},U<:AbstractArray{T,N}}
    lower::L
    upper::U
    function BoundedSet{T}(lower::L, upper::U) where {T,N,
                                                      L<:AbstractArray{T,N},
                                                      U<:AbstractArray{T,N}}
        isconcretetype(T) || throw(ArgumentError(
            "variables must have concrete element type, got $T"))
        T === float(T) || throw(ArgumentError(
            "variables must have floating-point element type, got $T"))
        axes(lower) == axes(upper) || throw(DimensionMismatch(pretty(
            "bounds must have the same axes, got ", axes(lower), " and ", axes(upper))))
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
