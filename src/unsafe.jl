# NOTE: dst !== x and axes(dst) == axes(x) must hold
unsafe_copy!(dst::AbstractArray, x::AbstractArray) = copyto!(dst, x)
unsafe_copy!(dst::DenseArray{T,N}, x::DenseArray{T,N}) where {T,N} = begin
    if isbitstype(T)
        nbytes = sizeof(T)*length(dst)
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), dst, x, nbytes)
    else
        copyto!(dst, x)
    end
    return dst
end

# The following structures are a trick to make our own closure objects to
# implement the `unsafe_ax!`, `unsafe_axpy!`, and `unsafe_axpby!` operations.
# This is needed to avoid *lots* of allocations (at least for `unsafe_axpby!`)
# and reach ultimate execution speed with `map!`.
struct αx{A} <: Function
    α::A
end
@inline (f::αx)(x) = f.α*x
αx(α::Real, x::AbstractArray) = αx(convert_multiplier(α, x))

struct αxpy{A} <: Function
    α::A
end
@inline (f::αxpy)(x, y) = f.α*x + y
αxpy(α::Real, x::AbstractArray) = αxpy(convert_multiplier(α, x))

struct αxmy{A} <: Function
    α::A
end
@inline (f::αxmy)(x, y) = f.α*x - y
αxmy(α::Real, x::AbstractArray) = αxpy(convert_multiplier(α, x))

struct αxpβy{A,B} <: Function
    α::A
    β::B
end
@inline (f::αxpβy)(x, y) = f.α*x + f.β*y
αxpβy(α::Real, x::AbstractArray, β::Real, y::AbstractArray) =
    αxpβy(convert_multiplier(α, x), convert_multiplier(β, y))

"""
    NumOptBase.unsafe_ax!(dst, α, x)

executes the low-level operation:

    dst[i] = α*x[i]

for all indices `i` of `dst` and `x` and assuming without checking that these
arguments have the same axes. The scalar `α` must not be zero and must have
been converted to a suitable floating-point type by the caller.

None of these assumptions are checked, this method is thus *unsafe* and shall
not be directly called but it may be extended for specific array types. By
default, it uses SIMD vectorization for strided arrays and calls
[`NumOptBase.unsafe_map!`](@ref) for other arrays.

"""
unsafe_ax!(dst::AbstractArray, α::Real, x::AbstractArray) =
    # FIXME: unsafe_map!(xᵢ -> α*xᵢ, dst, x)
    unsafe_map!(αx(α), dst, x)

"""
    NumOptBase.unsafe_axpy!(dst, α, x, y)

executes the low-level operation:

    dst[i] = α*x[i] + y[i]

for all indices `i` of `dst`, `x`, and `y` and assuming without checking that
these arguments have the same axes. The scalar `α` must not be zero and must
have been converted to a suitable floating-point type by the caller.

None of these assumptions are checked, this method is thus *unsafe* and shall
not be directly called but it may be extended for specific array types. By
default, it uses SIMD vectorization for strided arrays and calls
[`NumOptBase.unsafe_map!`](@ref) for other arrays.

"""
unsafe_axpy!(dst::AbstractArray, α::Real, x::AbstractArray, y::AbstractArray) =
    # FIXME: unsafe_map!((xᵢ, yᵢ) -> α*xᵢ + yᵢ, dst, x, y)
    unsafe_map!(αxpy(α), dst, x, y)

"""
    NumOptBase.unsafe_axpby!(dst, α, x, β, y)

executes the low-level operation:

    dst[i] = α*x[i] + β*y[i]

for all indices `i` of `dst`, `x`, and `y` and assuming without checking that
these arguments have the same axes. The scalars `α` and `β` must not be zero
and must have been converted to a suitable floating-point type by the caller.

None of these assumptions are checked, this method is thus *unsafe* and shall
not be directly called but it may be extended for specific array types. By
default, it uses SIMD vectorization for strided arrays and calls
[`NumOptBase.unsafe_map!`](@ref) for other arrays.

"""
unsafe_axpby!(dst::AbstractArray, α::Real, x::AbstractArray, β::Real, y::AbstractArray) =
    # FIXME: unsafe_map!((xᵢ, yᵢ) -> α*xᵢ + β*yᵢ, dst, x, y)
    unsafe_map!(αxpβy(α, β), dst, x, y)

"""
    NumOptBase.unsafe_map!(f, dst, args...)

overwrites `dst` with the result of applying `f` element-wise to `args...`.

This method may be extended for specific array types. By default, it uses SIMD
vectorization for strided arrays and calls `map!` for other arrays.

This method is *unsafe* because it assumes without checking that `dst` and all
`args...` have the same axes.

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
