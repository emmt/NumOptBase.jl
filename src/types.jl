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

* `SimdLoopEngine` - Julia loop without bound checking and with SIMD
  vectorization (`@inbounds @simd`);

* `TurboLoopEngine` - Julia loop without bound checking and with AVX
  vectorization (`@avx` or `@turbo`).

GPU arrays:

* `CudaEngine` - implementation suitable for `CuArray`.

Fall-back:

* `Engine`

"""
abstract type Engine end

"""
    NumOptBase.LoopEngine <: NumOptBase.Engine

is the abstract type identifying implementation with simple loops and bound
checking.

"""
abstract type LoopEngine <: Engine end

"""
    NumOptBase.InBoundsLoopEngine <: NumOptBase.LoopEngine

is the abstract type identifying implementation with simple in-bounds loops
(i.e. `@inbounds`).

"""
abstract type InBoundsLoopEngine <: LoopEngine end

"""
    NumOptBase.SimdLoopEngine <: NumOptBase.InBoundsLoopEngine

is the abstract type identifying implementation with `@simd` loops.

"""
abstract type SimdLoopEngine <: InBoundsLoopEngine end

"""
    NumOptBase.SimdLoopEngine <: NumOptBase.InBoundsLoopEngine

is the abstract type identifying implementation with `@avx` or `@turbo` loops.

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
