"""

Package `NumOptBase` provides basic operations on "variables" for numerical
optimization methods.

"""
module NumOptBase

using ArrayTools: @assert_same_axes
using Unitless: floating_point_type

"""
    NumOptBase.apply!(dst, f, args...) -> dst

overwrites destination `dst` with the result of applying the mapping `f` to
arguments `args...`.

This method shall be extended to specific argument types.

"""
apply!

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
    NumOptBase.update!(x, β, y) -> x

overwrites destination `x` with `x + β⋅y` and returns `x`. This is a shortcut
for `NumOptBase.combine!(x,1,x,β,y)`.

"""
function update!(dst::AbstractArray{T,N},
                 α::Real, x::AbstractArray{T,N}) where {T,N}
    @assert_same_axes dst x
    unsafe_update!(dst, α, x)
    return dst
end


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

This method may be extended for specific array types. By default, it uses
loop-vectorization for strided arrays and calls `map!` for other arrays.

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

yields the ℓ₁ norm of `x` considered as a *vector* (i.e; as if `x` has been
flattened).

"""
function norm1(x::AbstractArray)
    if x isa StridedArray
        acc = abs(zero(eltype(x)))
        @inbounds @simd for i in eachindex(x)
            acc += abs(x[i])
        end
        return acc
    else
        return mapreduce(abs, +, x)
    end
end

"""
    NumOptBase.norm2(x)

yields the Euclidean norm of `x` considered as a *vector* (i.e; as if `x` has
been flattened).

"""
function norm2(x::AbstractArray)
    if x isa StridedArray
        acc = abs2(zero(eltype(x)))
        @inbounds @simd for i in eachindex(x)
            acc += abs2(x[i])
        end
        return sqrt(acc)
    else
        v = flatten(x)
        return sqrt(v'*v)
    end
end

"""
    NumOptBase.norminf(x)

yields the infinite norm of `x` considered as a *vector* (i.e; as if `x` has
been flattened).

"""
function norminf(x::AbstractArray)
    if x isa StridedArray
        r = abs(zero(eltype(x)))
        @inbounds @simd for i in eachindex(x)
            a = abs(x[i])
            r = a > r ? a : r
        end
        return r
    else
        return reduce(max_abs, x; init = abs(zero(eltype(x))))
    end
end

max_abs(x, y) = (abs_y = abs(y)) > x ? abs_y : x

flatten(x::AbstractVector) = x
flatten(x::AbstractArray) = reshape(x, length(x))

end # module
