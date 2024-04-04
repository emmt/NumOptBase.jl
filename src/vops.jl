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
        unsafe_copy!(y, x)
    end
    return y
end

apply!(y::AbstractArray{T,N}, A::Diag{T,N}, x::AbstractArray{T,N}) where {T,N} =
    multiply!(y, diag(A), x)

LinearAlgebra.lmul!(dst::AbstractArray, A::Union{Diag,Identity}, b::AbstractArray) =
    apply!(dst, A, b)

"""
    NumOptBase.copy!(dst, src) -> dst

copies `src` into `dst` and returns `dst`. Arguments must have the same `axes`,
unlike `Base.copy!` which only requires this for multi-dimensional arrays and
resizes `dst` if needed when `src` and `dst` are vectors.

"""
function copy!(dst::AbstractArray, src::AbstractArray)
    if dst !== src
        @assert_same_axes dst src
        unsafe_copy!(dst, src)
    end
    return dst
end

"""
    unsafe_copy!(dst, src)

effectively copy `src` into `dst`. `dst !== src` and `axes(dst) == axes(src)`
both hold when this method is called. The fallback implementation simply calls
`copyto!(dst, src)`.

"""
unsafe_copy!(dst::AbstractArray, src::AbstractArray) = copyto!(dst, src)

"""
    NumOptBase.zerofill!(A) -> A

zero-fill `A` and returns it. The fallback implementation simply calls
`fill!(A, zero(eltype(A)))`.

"""
zerofill!(A::AbstractArray) = fill!(A, zero(eltype(A)))

"""
    NumOptBase.scale!([E,] dst, α, x) -> dst

overwrites destination `dst` with `α⋅x` and returns `dst`. If `iszero(α)`
holds, `dst` is zero-filled whatever the values in `x`.

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

overwrites `x` with `α⋅x` and returns `x`. If `iszero(α)` holds, `x` is
zero-filled whatever its contents.

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
optimized version of `NumOptBase.combine!(dst,1,dst,α,x)`. If `iszero(α)`
holds, `dst` is left unchanged whatever the values of `x`.

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
        unsafe_update!(E, dst, convert_multiplier(α, x), x)
        # FIXME: unsafe_map!(E, αxpy(α, x), dst, x, dst)
    end
    return dst
end

"""
    NumOptBase.update!([E,] dst, α, x, y) -> dst

overwrites destination `dst` with `dst + α⋅x⋅y` performed element-wise and
returns `dst`. If `iszero(α)` holds, `dst` is left unchanged whatever the
values of `x` and `y`.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, y)` is assumed.

"""
function update!(dst::AbstractArray{T,N},
                 α::Real, x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N}
    return update!(engine(dst, x, y), dst, α, x, y)
end

xpyz(x, y, z) = muladd(y, z, x) # x + y*z
xmyz(x, y, z) = x - y*z

function update!(::Type{E},
                 dst::AbstractArray{T,N},
                 α::Real, x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N,E<:Engine}
    @assert_same_axes dst x y
    if isone(α)
        unsafe_map!(E, xpyz, dst, dst, x, y)
    elseif α == -one(α)
        unsafe_map!(E, xmyz, dst, dst, x, y)
    elseif !iszero(α)
        unsafe_update!(E, dst, convert_multiplier(α, x), x, y)
    end
    return dst
end

for (optim, array, engine) in ((:none,      AbstractArray, LoopEngine),
                               (:inbounds,  AbstractArray, InBoundsLoopEngine),
                               (:simd,      SimdArray,     SimdLoopEngine))
    @eval begin
        @inline function unsafe_update!(::Type{<:$engine},
                                        dst::$array,
                                        α::Real,
                                        x::$array)
            @vectorize $optim for i in eachindex(dst, x)
                dst[i] += α*x[i]
            end
            return nothing
        end
        @inline function unsafe_update!(::Type{<:$engine},
                                        dst::$array,
                                        α::Real,
                                        x::$array,
                                        y::$array)
            @vectorize $optim for i in eachindex(dst, x, y)
                dst[i] += α*x[i]*y[i]
            end
            return nothing
        end
    end
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
    NumOptBase.combine!([E,] dst, x, ±, y) -> dst

overwrites destination `dst` with `x ± y` and returns `dst`. This is a shortcut
for:

    NumOptBase.combine!([E,] dst, 1, x, ±1, y) -> dst

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, y)` is assumed.

"""
function combine!(dst::AbstractArray{T,N}, x::AbstractArray{T,N},
                  pm::PlusOrMinus, y::AbstractArray{T,N}) where {T,N}
    return combine!(dst, 1, x, pm(1), y)
end

function combine!(::Type{E}, dst::AbstractArray{T,N}, x::AbstractArray{T,N},
                  pm::PlusOrMinus, y::AbstractArray{T,N}) where {T,N,E<:Engine}
    return combine!(E, dst, 1, x, pm(1), y)
end

"""
    NumOptBase.inner([T,] [E,] [w,] x, y)

yields the inner product of `x` and `y` computed as expected by numerical
optimization methods. If optional argument `w` is specified, `Σᵢ wᵢ⋅xᵢ⋅yᵢ` is
returned.

Optional arguments `T` and `E` can be given in any order. Optional argument `T`
specifies the type of the result. Optional argument `E` specifies which
*engine* to use for the computations. If unspecified, `E =
NumOptBase.engine([w,] x, y)` is assumed.

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
    NumOptBase.unsafe_inner(E, [w,] x, y)

executes the task of [`NumOptBase.inner`](@ref) assuming without checking that
array arguments have the same axes. This method is thus *unsafe* and shall not
be directly called but it may be extended for specific array types. By default,
it uses SIMD vectorization for strided arrays and calls `mapreduce` for other
arrays. Argument `E` specifies which *engine* to use for the computations.

""" unsafe_inner!

# Loop-based implementations.
for (optim, array, engine) in ((:none,      AbstractArray, LoopEngine),
                               (:inbounds,  AbstractArray, InBoundsLoopEngine),
                               (:simd,      SimdArray,     SimdLoopEngine))
    @eval begin
        @inline function unsafe_inner(::Type{<:$engine},
                                      x::$array,
                                      y::$array)
            acc = inner(zero(eltype(x)), zero(eltype(y)))
            @vectorize $optim for i in eachindex(x, y)
                acc += inner(x[i], y[i])
            end
            return acc
        end
        @inline function unsafe_inner(::Type{<:$engine},
                                      w::$array,
                                      x::$array,
                                      y::$array)
            acc = inner(zero(eltype(w)), zero(eltype(x)), zero(eltype(y)))
            @vectorize $optim for i in eachindex(w, x, y)
                acc += inner(w[i], x[i], y[i])
            end
            return acc
        end
    end
end

# Generic implementations based on `mapreduce`.
@inline function unsafe_inner(::Type{<:Engine},
                              x::AbstractArray,
                              y::AbstractArray)
    return mapreduce(inner, +, x, y)
end
@inline function unsafe_inner(::Type{<:Engine},
                              w::AbstractArray,
                              x::AbstractArray,
                              y::AbstractArray)
    return mapreduce(inner, +, w, x, y)
end

"""
    NumOptBase.norm1([T,] [E,] x)

yields the ℓ₁ norm of `x` considered as a real-valued *vector* (i.e., as if `x`
has been flattened).

If `x` is an array, two optional arguments `T` and `E` can be specified in any
order. Optional argument `T` specifies the type of the result. Optional
argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(x)` is assumed.

"""
norm1(x::Real) = abs(x)
norm1(x::Complex{<:Real}) = abs(real(x)) + abs(imag(x))
norm1(x::AbstractArray) = norm1(engine(x), x)

# Loop-based implementations.
for (optim, array, engine) in ((:none,      AbstractArray, LoopEngine),
                               (:inbounds,  AbstractArray, InBoundsLoopEngine),
                               (:simd,      SimdArray,     SimdLoopEngine))
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
    NumOptBase.norm2([T,] [E,] x)

yields the Euclidean norm of `x` considered as a real-valued *vector* (i.e., as
if `x` has been flattened).

If `x` is an array, two optional arguments `T` and `E` can be specified in any
order. Optional argument `T` specifies the type of the result. Optional
argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(x)` is assumed.

"""
norm2(x::Real) = abs(x)
norm2(x::Complex{<:Real}) = sqrt(abs2(x))
norm2(x::AbstractArray) = norm2(engine(x), x)

# Loop-based implementations.
for (optim, array, engine) in ((:none,      AbstractArray, LoopEngine),
                               (:inbounds,  AbstractArray, InBoundsLoopEngine),
                               (:simd,      SimdArray,     SimdLoopEngine))
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
    NumOptBase.norminf([T,] [E,] x)

yields the infinite norm of `x` considered as a real-valued *vector* (i.e., as
if `x` has been flattened).

If `x` is an array, two optional arguments `T` and `E` can be specified in any
order. Optional argument `T` specifies the type of the result. Optional
argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(x)` is assumed.

"""
norminf(x::Real) = abs(x)
norminf(x::Complex{<:Real}) = max(abs(real(x)), abs(imag(x)))
norminf(x::AbstractArray) = norminf(engine(x), x)

# Generic implementation based on `mapreduce`.
norminf(::Type{<:Engine}, x::AbstractArray) = mapreduce(norminf, max, x)

# Implement floating-point type conversion for norms and inner prodcut.
for func in (:norm1, :norm2, :norminf)
    @eval begin
        $func(::Type{T}, x::AbstractArray) where {T<:Number} =
            as(T, $func(x))
        $func(::Type{T}, ::Type{E}, x::AbstractArray) where {T<:Number,E<:Engine} =
            as(T, $func(E, x))
        $func(::Type{E}, ::Type{T}, x::AbstractArray) where {T<:Number,E<:Engine} =
            as(T, $func(E, x))
    end
end
@inline inner(::Type{T}, x::AbstractArray...) where {T<:Number} =
    as(T, inner(x...))
@inline inner(::Type{T}, ::Type{E}, x::AbstractArray...) where {T<:Number,E<:Engine} =
    as(T, inner(E, x...))
@inline inner(::Type{E}, ::Type{T}, x::AbstractArray...) where {T<:Number,E<:Engine} =
    as(T, inner(E, x...))
