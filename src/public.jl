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
    NumOptBase.copy!(dst, x) -> dst

copies `x` into `dst` and returns `dst`.

"""
function copy!(dst::AbstractArray, x::AbstractArray)
    if dst !== x
        @assert_same_axes dst x
        unsafe_copy!(dst, x)
    end
    return dst
end

"""
    NumOptBase.zerofill!(dst) -> dst

zero-fill `dst` and returns it.

"""
zerofill!(dst::AbstractArray{T}) where {T} = fill!(dst, zero(T))
function zerofill!(dst::DenseArray{T}) where {T}
    if isbitstype(T)
        nbytes = sizeof(T)*length(dst)
        @ccall memset(dst::Ptr{Cvoid}, 0::Cint, nbytes::Csize_t)::Ptr{Cvoid}
    else
        fill!(dst, zero(T))
    end
    return dst
end

"""
    NumOptBase.scale!(dst, α, x) -> dst

overwrites destination `dst` with `α⋅x` and returns `dst`. If `iszero(α)`
holds, zero-fill `dst` whatever the values in `x`.

"""
function scale!(dst::AbstractArray{T,N},
                α::Real, x::AbstractArray{T,N}) where {T,N}
    @assert_same_axes dst x
    if iszero(α)
        zerofill!(dst)
    elseif isone(α)
        dst !== x && unsafe_copy!(dst, x)
    elseif α == -one(α)
        unsafe_map!(-, dst, x)
    else
        unsafe_map!(αx(α, x), dst, x)
    end
    return dst
end

"""
    NumOptBase.update!(dst, α, x) -> dst

overwrites destination `dst` with `dst + α⋅x` and returns `dst`. This is a shortcut
for `NumOptBase.combine!(dst,1,dst,α,x)`.

"""
function update!(dst::AbstractArray{T,N},
                 α::Real, x::AbstractArray{T,N}) where {T,N}
    @assert_same_axes dst x
    if isone(α)
        unsafe_map!(+, dst, dst, x)
    elseif α == -one(α)
        unsafe_map!(-, dst, dst, x)
    elseif !iszero(α)
        unsafe_map!(αxpy(α, x), dst, x, dst)
        #unsafe_map!((dstᵢ, xᵢ) -> dstᵢ + α*xᵢ, dst, dst, x)
    end
    return dst
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
    if iszero(β)
        # NOTE: Like scale!(dst, α, x)
        if iszero(α)
            zerofill!(dst)
        elseif isone(α)
            dst !== x && unsafe_copy!(dst, x)
        elseif α == -one(α)
            unsafe_map!(-, dst, x)
        else
            unsafe_map!(αx(α, x), dst, x)
        end
    elseif iszero(α)
        # NOTE: Like scale!(dst, β, y) except β is not zero
        if isone(β)
            dst !== y && unsafe_copy!(dst, y)
        elseif β == -one(β)
            unsafe_map!(-, dst, y)
        else
            unsafe_map!(αx(β, y), dst, y)
        end
    elseif isone(α)
        if isone(β)
            # dst .= x .+ y
            unsafe_map!(+, dst, x, y)
        elseif β == -one(β)
            # dst .= x .- y
            unsafe_map!(-, dst, x, y)
        else
            # dst .= x .+ β.*y
            unsafe_map!(αxpy(β, y), dst, y, x)
        end
    else
        # dst .= α.*x .+ β.*y
        unsafe_map!(αxpβy(α, x, β, y), dst, x, y)
    end
    return dst
end

"""
    NumOptBase.convert_multiplier(α, x)

converts scalar real `α` to a floating-point type whose numerical precision is
the same as that of the elements of `x`.

"""
convert_multiplier(α::Real, x::AbstractArray) = as(floating_point_type(x), α)

"""
    NumOptBase.as(T, x)

converts `x` to type `T`. The result is type-asserted to be of type `T`. If `x
isa T` holds, `x` is returned unchanged.

"""
as(::Type{T}, x::T) where {T} = x
as(::Type{T}, x) where {T} = convert(T, x)::T

"""
    NumOptBase.floating_point_type(x)

yields the floating-point type corresponding to the numeric type or value `x`.
If `x` is a numeric array type or instance, the floating-point type of the
elements of `x` is returned. If `x` is complex, the floating-point type of the
real and imaginary parts of `x` is returned.

"""
floating_point_type(x::Any) = floating_point_type(typeof(x))
floating_point_type(::Type{T}) where {T<:AbstractArray} = floating_point_type(eltype(T))
floating_point_type(::Type{T}) where {R<:Real,T<:RealComplex{R}} = float(R)
@noinline floating_point_type(T::Union{DataType,UnionAll}) =
    throw(ArgumentError("cannot determine floating-point type of `$T`"))

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
inner(w::Real, x::Real, y::Real) = w*x*y

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
