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
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), dst, 0, nbytes)
    else
        fill!(dst, zero(T))
    end
    return dst
end

"""
    NumOptBase.scale!([E,] dst, α, x) -> dst

overwrites destination `dst` with `α⋅x` and returns `dst`. If `iszero(α)`
holds, zero-fill `dst` whatever the values in `x`.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x)` is assumed.

"""
function scale!(dst::AbstractArray{T,N},
                α::Real, x::AbstractArray{T,N}) where {T,N}
    return scale!(engine(dst, x), dst, α, x)
end

function scale!(::Type{E},
                dst::AbstractArray{T,N},
                α::Real, x::AbstractArray{T,N}) where {T,N,E<:Engine}
    @assert_same_axes dst x
    if iszero(α)
        zerofill!(dst)
    elseif isone(α)
        dst !== x && unsafe_copy!(dst, x)
    elseif α == -one(α)
        unsafe_map!(E, -, dst, x)
    else
        unsafe_map!(E, αx(α, x), dst, x)
    end
    return dst
end

"""
    NumOptBase.scale!([E,] α, x) -> x
    NumOptBase.scale!([E,] x, α) -> x

overwrites `x` with `α⋅x` and returns `x`. If `iszero(α)` holds, zero-fill `x`
whatever its contents.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(x)` is assumed.

"""
scale!(α::Real, x::AbstractArray) = scale!(x, α)
scale!(x::AbstractArray, α::Real) = scale!(engine(x), x, α)
scale!(::Type{E}, α::Real, x::AbstractArray) where {E<:Engine} = scale!(E, x, α)
function scale!(::Type{E}, x::AbstractArray, α::Real) where {E<:Engine}
    if iszero(α)
        zerofill!(x)
    elseif α == -one(α)
        unsafe_map!(E, -, x, x)
    elseif !isone(α)
        unsafe_map!(E, αx(α, x), x, x)
    end
    return x
end

"""
    NumOptBase.update!([E,] dst, α, x) -> dst

overwrites destination `dst` with `dst + α⋅x` and returns `dst`. This is an
optimized version of `NumOptBase.combine!(dst,1,dst,α,x)`.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x)` is assumed.

"""
function update!(dst::AbstractArray{T,N},
                 α::Real, x::AbstractArray{T,N}) where {T,N}
    return update!(engine(dst, x), dst, α, x)
end

function update!(::Type{E},
                 dst::AbstractArray{T,N},
                 α::Real, x::AbstractArray{T,N}) where {T,N,E<:Engine}
    @assert_same_axes dst x
    if isone(α)
        unsafe_map!(E, +, dst, dst, x)
    elseif α == -one(α)
        unsafe_map!(E, -, dst, dst, x)
    elseif !iszero(α)
        unsafe_map!(E, αxpy(α, x), dst, x, dst)
    end
    return dst
end

"""
    NumOptBase.multiply!([E,] dst, x, y) -> dst

overwrites destination `dst` with the element-wise multiplication (Hadamard
product) of `x` by `y` and returns `dst`.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, y)` is assumed.

"""
function multiply!(dst::AbstractArray{T,N},
                   x::AbstractArray{T,N},
                   y::AbstractArray{T,N}) where {T,N}
    return multiply!(engine(dst, x, y), dst, x, y)
end

function multiply!(::Type{E},
                   dst::AbstractArray{T,N},
                   x::AbstractArray{T,N},
                   y::AbstractArray{T,N}) where {T,N,E<:Engine}
    @assert_same_axes dst x y
    unsafe_map!(E, *, dst, x, y)
    return dst
end

"""
    NumOptBase.combine!([E,] dst, α, x, β, y) -> dst

overwrites destination `dst` with the linear combination `α⋅x + β⋅y` and
returns `dst`.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, y)` is assumed.

"""
function combine!(dst::AbstractArray{T,N},
                  α::Real, x::AbstractArray{T,N},
                  β::Real, y::AbstractArray{T,N}) where {T,N}
    return combine!(engine(dst, x, y), dst, α, x, β, y)
end

function combine!(::Type{E},
                  dst::AbstractArray{T,N},
                  α::Real, x::AbstractArray{T,N},
                  β::Real, y::AbstractArray{T,N}) where {T,N,E<:Engine}
    @assert_same_axes dst x y
    if iszero(β)
        # NOTE: Like scale!(dst, α, x)
        if iszero(α)
            zerofill!(dst)
        elseif isone(α)
            dst !== x && unsafe_copy!(dst, x)
        elseif α == -one(α)
            unsafe_map!(E, -, dst, x)
        else
            unsafe_map!(E, αx(α, x), dst, x)
        end
    elseif iszero(α)
        # NOTE: Like scale!(dst, β, y) except β is not zero
        if isone(β)
            dst !== y && unsafe_copy!(dst, y)
        elseif β == -one(β)
            unsafe_map!(E, -, dst, y)
        else
            unsafe_map!(E, αx(β, y), dst, y)
        end
    elseif isone(α)
        if isone(β)
            # dst .= x .+ y
            unsafe_map!(E, +, dst, x, y)
        elseif β == -one(β)
            # dst .= x .- y
            unsafe_map!(E, -, dst, x, y)
        else
            # dst .= x .+ β.*y
            unsafe_map!(E, αxpy(β, y), dst, y, x)
        end
    else
        # dst .= α.*x .+ β.*y
        unsafe_map!(E, αxpβy(α, x, β, y), dst, x, y)
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
    NumOptBase.inner([E,] [w,] x, y)

yields the inner product of `x` and `y` computed as expected by numerical
optimization methods. If optional argument `w` is specified, `Σᵢ wᵢ⋅xᵢ⋅yᵢ` is
returned.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine([w,] x, y)` is assumed.

"""
inner(x::AbstractArray, y::AbstractArray) = inner(engine(x, y), x, y)
inner(w::AbstractArray, x::AbstractArray, y::AbstractArray) =
    inner(engine(w, x, y), w, x, y)

function inner(::Type{E}, x::AbstractArray, y::AbstractArray) where {E<:Engine}
    @assert_same_axes x y
    return unsafe_inner(E, x, y)
end

function inner(::Type{E}, w::AbstractArray, x::AbstractArray, y::AbstractArray) where {E<:Engine}
    @assert_same_axes w x y
    return unsafe_inner(E, w, x, y)
end

# Inner product of variables as assumed for numerical optimization, that is
# considering complexes as pairs of reals.
inner(x::Real, y::Real) = x*y
inner(x::Complex, y::Complex) = real(x)*real(y) + imag(x)*imag(y)
inner(w::Real, x::Real, y::Real) = w*x*y

"""
    NumOptBase.norm1([E,] x)

yields the ℓ₁ norm of `x` considered as a real-valued *vector* (i.e, as if `x`
has been flattened).

If `x` is an array, optional argument `E` specifies which *engine* to use for
the computations. If unspecified, `E = NumOptBase.engine(x)` is assumed.

"""
norm1(x::Real) = abs(x)
norm1(x::Complex{<:Real}) = abs(real(x)) + abs(imag(x))
norm1(x::AbstractArray) = norm1(engine(x), x)

# Loop-based implementations.
for (optim, array, engine) in ((:none,      AbstractArray, LoopEngine),
                               (:inbounds,  AbstractArray, InBoundsLoopEngine),
                               (:simd,      StridedArray,  SimdLoopEngine))
    @eval begin
        @inline function norm1(::Type{<:$engine}, x::$array)
            acc = norm1(zero(eltype(x)))
            @vectorize $optim for i in eachindex(x)
                acc += norm1(x[i])
            end
            return acc
        end
    end
end

# Generic implementation based on `mapreduce`.
norm1(::Type{<:Engine}, x::AbstractArray) = mapreduce(norm1, +, x)

"""
    NumOptBase.norm2([E,] x)

yields the Euclidean norm of `x` considered as a real-valued *vector* (i.e, as
if `x` has been flattened).

If `x` is an array, optional argument `E` specifies which *engine* to use for
the computations. If unspecified, `E = NumOptBase.engine(x)` is assumed.

"""
norm2(x::Real) = abs(x)
norm2(x::Complex{<:Real}) = sqrt(abs2(x))
norm2(x::AbstractArray) = norm2(engine(x), x)

# Loop-based implementations.
for (optim, array, engine) in ((:none,      AbstractArray, LoopEngine),
                               (:inbounds,  AbstractArray, InBoundsLoopEngine),
                               (:simd,      StridedArray,  SimdLoopEngine))
    @eval begin
        @inline function norm2(::Type{<:$engine}, x::$array)
            acc = abs2(zero(eltype(x)))
            @vectorize $optim for i in eachindex(x)
                acc += abs2(x[i])
            end
            return sqrt(acc)
        end
    end
end

# Generic implementation based on `mapreduce`.
#
# NOTE: Base.abs2 does this:
#     abs2(x::Real) = x*x
#     abs2(x::Complex) = abs2(real(x)) + abs2(imag(x))
norm2(::Type{<:Engine}, x::AbstractArray) = sqrt(mapreduce(abs2, +, x))

"""
    NumOptBase.norminf([E,] x)

yields the infinite norm of `x` considered as a real-valued *vector* (i.e, as
if `x` has been flattened).

If `x` is an array, optional argument `E` specifies which *engine* to use for
the computations. If unspecified, `E = NumOptBase.engine(x)` is assumed.

"""
norminf(x::Real) = abs(x)
norminf(x::Complex{<:Real}) = max(abs(real(x)), abs(imag(x)))
norminf(x::AbstractArray) = norminf(engine(x), x)

# Generic implementation based on `mapreduce`.
norminf(::Type{<:Engine}, x::AbstractArray) = mapreduce(norminf, max, x)

flatten(x::AbstractVector) = x
flatten(x::AbstractArray) = reshape(x, length(x))
