"""

Package `NumOptBase` provides basic operations on "variables" for numerical
optimization methods.

"""
module NumOptBase

using ArrayTools: @assert_same_axes
using Unitless: floating_point_type
using LinearAlgebra

if !isdefined(Base, :get_extension)
    using Requires
end

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
end
LinearAlgebra.diag(A::Diag) = A.diag

"""
    NumOptBase.apply!(dst, f, args...) -> dst

overwrites destination `dst` with the result of applying the mapping `f` to
arguments `args...`.

As implemented in `NumOptBase`, this method only handles a few types of
mappings:

- If `f` is an array, a generalized matrix-vector multiplication is applied.

- If `f` is [`NumOptBase.Identity()`](@ref), the identity mapping is applied.
  The constant [`NumOptBase.Id`](@ref) is the singleton object of type
  [`NumOptBase.Identity`](@ref).

- If `f` is an instance of [`NumOptBase.Diag`](@ref), an element-wise
  multiplication by `diag(f)` is applied.

The `NumOptBase.apply!` method shall be specialized in other argument types to
handle other cases.

"""
function apply!(y::AbstractArray{T,Ny},
                A::AbstractArray{T,Na},
                x::AbstractArray{T,Nx}) where {T<:Real,Ny,Na,Nx}
    Na == Nx + Ny || throw(DimensionMismatch("incompatible number of dimensions"))
    inds = axes(A)
    (Ny ≤ Na && axes(y) == inds[1:Ny]) || throw(DimensionMismatch(
        "axes of output array must be the same as the leading axes of the \"matrix\""))
    (Nx ≤ Na && axes(x) == inds[Na-Nx+1:Na]) || throw(DimensionMismatch(
        "axes of input array must be the same as the trailing axes of the \"matrix\""))
    if Ny == 1 && Nx == 1
        mul!(y, A, x)
    else
        mul!(flatten(y), reshape(A, (length(y), length(x))), flatten(x))
    end
    return y
end

function apply!(y::AbstractArray{T,N}, ::Identity, x::AbstractArray{T,N}) where {T,N}
    if y !== x
        @assert_same_axes x y
        copyto!(y, x)
    end
    return y
end

apply!(y::AbstractArray{T,N}, A::Diag{T,N}, x::AbstractArray{T,N}) where {T,N} =
    multiply!(y, diag(A), x)

"""
    NumOptBase.scale!(dst, α, x) -> dst

overwrites destination `dst` with `α⋅x` and returns `dst`. If `iszero(α)`
holds, zero-fill `dst` whatever the values in `x`.

"""
function scale!(dst::AbstractArray{T,N},
                α::Real, x::AbstractArray{T,N}) where {T,N}
    @assert_same_axes dst x
    unsafe_scale!(dst, α, x)
    return dst
end

"""
    NumOptBase.unsafe_scale!(dst, α, x)

executes the task of [`NumOptBase.scale!`](@ref) assuming without checking
that array arguments have the same axes. This method is thus *unsafe* and shall
not be directly called but it may be extended for specific array types. By
default, it uses SIMD vectorization for strided arrays and calls `map!` for
other arrays.

"""
function unsafe_scale!(dst::AbstractArray,
                       α::Real, x::AbstractArray)
    if iszero(α)
        fill!(dst, zero(eltype(dst)))
    elseif isone(α)
        dst !== x && copyto!(dst, x)
    elseif α == -one(α)
        unsafe_map!(-, dst, x)
    else
        α = convert_multiplier(α, x)
        unsafe_map!(xᵢ -> α*xᵢ, dst, x)
    end
    nothing
end

"""
    NumOptBase.update!(dst, α, x) -> dst

overwrites destination `dst` with `dst + α⋅x` and returns `dst`. This is a shortcut
for `NumOptBase.combine!(dst,1,dst,α,x)`.

"""
function update!(dst::AbstractArray{T,N},
                 α::Real, x::AbstractArray{T,N}) where {T,N}
    @assert_same_axes dst x
    unsafe_update!(dst, α, x)
    return dst
end

"""
    NumOptBase.unsafe_update!(dst, α, x)

executes the task of [`NumOptBase.update!`](@ref) assuming without checking
that array arguments have the same axes. This method is thus *unsafe* and shall
not be directly called but it may be extended for specific array types. By
default, it calls [`NumOptBase.unsafe_map!`](@ref).

"""
function unsafe_update!(dst::AbstractArray,
                        α::Real, x::AbstractArray)
    if isone(α)
        unsafe_map!(+, dst, dst, x)
    elseif α == -one(α)
        unsafe_map!(-, dst, dst, x)
    elseif !iszero(α)
        α = convert_multiplier(α, x)
        unsafe_map!((dstᵢ, xᵢ) -> dstᵢ + α*xᵢ, dst, dst, x)
    end
    nothing
end

"""
    NumOptBase.multiply!(dst, x, y) -> dst

overwrites destination `dst` with the element-wise multiplication (Hadamard
product) of `x` by `y` and returns `dst`.

"""
function multiply!(dst::AbstractArray{T,N},
                   x::AbstractArray{T,N},
                   y::AbstractArray{T,N}) where {T,N}
    @assert_same_axes dst x y
    unsafe_map!(*, dst, x, y)
    return dst
end

"""
    NumOptBase.combine!(dst, α, x, β, y) -> dst

overwrites destination `dst` with the linear combination `α⋅x + β⋅y` and
returns `dst`.

"""
function combine!(dst::AbstractArray{T,N},
                  α::Real, x::AbstractArray{T,N},
                  β::Real, y::AbstractArray{T,N}) where {T,N}
    @assert_same_axes dst x y
    unsafe_combine!(dst, α, x, β, y)
    return dst
end

# The following structure is a trick to make our own closure object to
# implement the `combine!` operation. This is needed to avoid *lots* of
# allocations and reach ultimate execution speed.
struct Combine{F,A,B} <: Function
    f::F
    α::A
    β::B
end
@inline (op::Combine)(x, y) = op.f(op.α, x, op.β, y)
@inline combine_x(α, x, β, y) = x
@inline combine_y(α, x, β, y) = y
@inline combine_ax(α, x, β, y) = α*x
@inline combine_by(α, x, β, y) = β*y
@inline combine_xpy(α, x, β, y) = x + y
@inline combine_xmy(α, x, β, y) = x - y
@inline combine_xpby(α, x, β, y) = x + β*y
@inline combine_axpby(α, x, β, y) = α*x + β*y

"""
    NumOptBase.unsafe_combine!(dst, α, x, β, y)

executes the task of [`NumOptBase.combine!`](@ref) assuming without checking
that array arguments have the same axes. This method is thus *unsafe* and shall
not be directly called but it may be extended for specific array types. By
default, it calls [`NumOptBase.unsafe_map!`](@ref) with a special callable
object to perform the operation for a single variable.

"""
function unsafe_combine!(dst::AbstractArray,
                         α::Real, x::AbstractArray,
                         β::Real, y::AbstractArray)
    if iszero(β)
        unsafe_scale!(dst, α, x)
    elseif iszero(α)
        unsafe_scale!(dst, β, y)
    elseif isone(α)
        if isone(β)
            # dst .= x .+ y
            unsafe_map!(+, dst, x, y)
        elseif β == -one(β)
            # dst .= x .- y
            unsafe_map!(-, dst, x, y)
        else
            # dst .= x .+ β.*y
            unsafe_map!(
                Combine(combine_xpby, 1, convert_multiplier(β, y)),
                dst, x, y)
        end
    else
        # dst .= α.*x .+ β.*y
        unsafe_map!(
            Combine(combine_axpby, convert_multiplier(α, x), convert_multiplier(β, y)),
            dst, x, y)
    end
    nothing
end

"""
    NumOptBase.convert_multiplier(α, x)

converts scalar real `α` to a floating-point type whose numerical precision is
the same as that of the elements of `x`.

"""
convert_multiplier(α::Real, x::AbstractArray) =
    convert(floating_point_type(eltype(x)), α)

"""
    NumOptBase.unsafe_map!(f, dst, args...)

overwrites `dst` with the result of applying `f` element-wise to `args...`.

This method may be extended for specific array types. By default, it uses SIMD
vectorization for strided arrays and calls `map!` for other arrays.

This method is *unsafe* because it assumes without checking that `dst` and all
`args...` have the same indices.

"""
@inline function unsafe_map!(f::Function,
                             dst::AbstractArray,
                             x::AbstractArray)
    if dst isa StridedArray && x isa StridedArray
        @inbounds @simd for i in eachindex(dst, x)
            dst[i] = f(x[i])
        end
    else
        map!(f, dst, x)
    end
    nothing
end

@inline function unsafe_map!(f::Function,
                             dst::AbstractArray,
                             x::AbstractArray,
                             y::AbstractArray)
    if dst isa StridedArray && x isa StridedArray && y isa StridedArray
        @inbounds @simd for i in eachindex(dst, x, y)
            dst[i] = f(x[i], y[i])
        end
    else
        map!(f, dst, x, y)
    end
    nothing
end

"""
    NumOptBase.inner([w,] x, y)

yields the inner product of `x` and `y` computed as expected by numerical
optimization methods.

"""
function inner(x::AbstractArray{<:Number,N},
               y::AbstractArray{<:Number,N}) where {N}
    @assert_same_axes x y
    return unsafe_inner(x, y)
end

function inner(w::AbstractArray{<:Real,N},
               x::AbstractArray{<:Real,N},
               y::AbstractArray{<:Number,N}) where {N}
    @assert_same_axes w x y
    return unsafe_inner(w, x, y)
end

# Inner product of variables as assumed for numerical optimization, that is
# considering complexes as pairs of reals.
inner(x::Real, y::Real) = x*y
inner(x::Complex, y::Complex) = real(x)*real(y) + imag(x)*imag(y)

"""
    NumOptBase.unsafe_inner!([w,] x, y)

executes the task of [`NumOptBase.inner!`](@ref) assuming without checking that
array arguments have the same axes. This method is thus *unsafe* and shall not
be directly called but it may be extended for specific array types. By default,
it uses SIMD vectorization for strided arrays and calls `mapreduce` for other
arrays.

"""
function unsafe_inner(x::AbstractArray,
                      y::AbstractArray)
    if x isa StridedArray && y isa StridedArray
        acc = inner(zero(eltype(x)), zero(eltype(y)))
        @inbounds @simd for i in eachindex(x, y)
            acc += inner(x[i], y[i])
        end
        return acc
    else
        return mapreduce(inner, +, x, y)
    end
end

inner(w::Real, x::Real, y::Real) = w*x*y

function unsafe_inner(w::AbstractArray,
                      x::AbstractArray,
                      y::AbstractArray)
     if w isa StridedArray && x isa StridedArray && y isa StridedArray
         acc = inner(zero(eltype(w)), zero(eltype(x)), zero(eltype(y)))
         @inbounds @simd for i in eachindex(w, x, y)
             acc += inner(w[i], x[i], y[i])
         end
         return acc
     else
         return mapreduce(inner, +, w, x, y)
     end
end

"""
    NumOptBase.norm1(x)

yields the ℓ₁ norm of `x` considered as a real-valued *vector* (i.e, as if `x`
has been flattened).

"""
norm1(x::Real) = abs(x)
norm1(x::Complex{<:Real}) = abs(real(x)) + abs(imag(x))
function norm1(x::AbstractArray)
    if x isa StridedArray
        acc = norm1(zero(eltype(x)))
        @inbounds @simd for i in eachindex(x)
            acc += norm1(x[i])
        end
        return acc
    else
        return mapreduce(norm1, +, x)
    end
end

"""
    NumOptBase.norm2(x)

yields the Euclidean norm of `x` considered as a real-valued *vector* (i.e, as
if `x` has been flattened).

"""
norm2(x::Real) = abs(x)
norm2(x::Complex{<:Real}) = sqrt(abs2(x))
function norm2(x::AbstractArray)
    if x isa StridedArray
        acc = abs2(zero(eltype(x)))
        @inbounds @simd for i in eachindex(x)
            acc += abs2(x[i])
        end
        return sqrt(acc)
    else
        return sqrt(mapreduce(abs2, +, x))
    end
end
# NOTE: Base.abs2 does this:
#     abs2(x::Real) = x*x
#     abs2(x::Complex) = abs2(real(x)) + abs2(imag(x))

"""
    NumOptBase.norminf(x)

yields the infinite norm of `x` considered as a real-valued *vector* (i.e, as
if `x` has been flattened).

"""
norminf(x::Real) = abs(x)
norminf(x::Complex{<:Real}) = max(abs(real(x)), abs(imag(x)))
norminf(x::AbstractArray) = mapreduce(norminf, max, x)

flatten(x::AbstractVector) = x
flatten(x::AbstractArray) = reshape(x, length(x))

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include(
            "../ext/NumOptBaseCUDAExt.jl")
        @require LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890" include(
            "../ext/NumOptBaseLoopVectorizationExt.jl")
    end
end

end # module
