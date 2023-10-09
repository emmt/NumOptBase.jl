"""
    NumOptBase.engine(args...) -> E::Type{<:NumOptBase.Engine}

yields the type of the implementation of numerical operations for array
arguments `args...`.

"""
engine() = Engine
@inline engine(::StridedArray...) = TurboLoopEngine
@inline engine(::AbstractArray...) = Engine

"""
    NumOptBase.@vectorize optim for ...

compiles the `for ...` loop according to optimization `optim`, one of:

    :none     # no optimization, does bound checking
    :inbounds # optimize with @inbounds
    :simd     # optimize with @inbounds @simd
    :avx      # optimize with @avx (requires LoopVectorization)
    :turbo    # optimize with @turbo (requires LoopVectorization)

"""
macro vectorize(optim::Symbol, loop::Expr)
    loop.head === :for || error("expecting a `for` loop argument to `@vectorize`")
    esc(_vectorize(optim, loop))
end

_vectorize(optim::Symbol, loop::Expr) =
    optim === :none     ? loop :
    optim === :inbounds ? :(@inbounds $loop) :
    optim === :simd     ? :(@inbounds @simd $loop) :
    optim === :avx      ? :(@avx $loop) :
    optim === :turbo    ? :(@turbo $loop) :
    error("unknown loop optimizer `:$optim`")

"""
    NumOptBase.flatten(x)

converts array `x` in to a vector.

"""
flatten(x::AbstractVector) = x
flatten(x::AbstractArray) = reshape(x, length(x))

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

# The structures `αx`, `αxpy`, `αxmy`, and `αxpβy` are a trick to make our own
# closure objects to implement the `unsafe_ax!`, `unsafe_axpy!`, and
# `unsafe_axpby!` operations. This is needed to avoid *lots* of allocations
# when using closures and anonymous functions (at least for `αxpβy`) and reach
# ultimate execution speed with `map!`.

"""

For a scalar real `α` and an array `x`:

    f = NumOptBase.αx(α, x)

yields a callable object such that:

    f(xᵢ) -> α*xᵢ

with `α` converted to a floating-point type suitable for multiplication by the
elements of `x` (see [`NumOptBase.convert_multiplier`](@ref)). The object `f`
may be used with [`NumOptBase.unsafe_map!`](@ref).

"""
struct αx{A} <: Function
    α::A
end
@inline (f::αx)(x) = f.α*x
αx(α::Real, x::AbstractArray) = αx(convert_multiplier(α, x))

"""

For a scalar real `α` and an array `x`:

    f = NumOptBase.αxpy(α, x)

yields a callable object such that:

    f(xᵢ, yᵢ) -> α*xᵢ + yᵢ

with `α` converted to a floating-point type suitable for multiplication by the
elements of `x` (see [`NumOptBase.convert_multiplier`](@ref)). The object `f`
may be used with [`NumOptBase.unsafe_map!`](@ref).

"""
struct αxpy{A} <: Function
    α::A
end
@inline (f::αxpy)(x, y) = f.α*x + y
αxpy(α::Real, x::AbstractArray) = αxpy(convert_multiplier(α, x))

"""

For a scalar real `α` and an array `x`:

    f = NumOptBase.αxmy(α, x)

yields a callable object such that:

    f(xᵢ, yᵢ) -> α*xᵢ - yᵢ

with `α` converted to a floating-point type suitable for multiplication by the
elements of `x` (see [`NumOptBase.convert_multiplier`](@ref)). The object `f`
may be used with [`NumOptBase.unsafe_map!`](@ref).

"""
struct αxmy{A} <: Function
    α::A
end
@inline (f::αxmy)(x, y) = f.α*x - y
αxmy(α::Real, x::AbstractArray) = αxpy(convert_multiplier(α, x))

"""

For scalar reals `α` and `β` and arrays `x` and `y`:

    f = NumOptBase.αxpβy(α, x, β, y)

yields a callable object such that:

    f(xᵢ, yᵢ) -> α*xᵢ + β*yᵢ

with `α` and `β` converted to a floating-point type suitable for respective
multiplication by the elements of `x` and `y` (see
[`NumOptBase.convert_multiplier`](@ref)). The object `f` may be used with
[`NumOptBase.unsafe_map!`](@ref).

"""
struct αxpβy{A,B} <: Function
    α::A
    β::B
end
@inline (f::αxpβy)(x, y) = f.α*x + f.β*y
αxpβy(α::Real, x::AbstractArray, β::Real, y::AbstractArray) =
    αxpβy(convert_multiplier(α, x), convert_multiplier(β, y))

"""
    NumOptBase.unsafe_map!(E, f, dst, args...)

overwrites `dst` with the result of applying `f` element-wise to `args...`.

This method may be extended for specific array types. By default, it uses SIMD
vectorization for strided arrays and calls `map!` for other arrays.

This method is *unsafe* because it assumes without checking that `dst` and all
`args...` have the same axes.

Argument `E` specifies which *engine* to be use for the computations.

""" unsafe_map!

# Loop-based implementations.
for (optim, array, engine) in ((:none,      AbstractArray, LoopEngine),
                               (:inbounds,  AbstractArray, InBoundsLoopEngine),
                               (:simd,      StridedArray,  SimdLoopEngine))
    @eval begin
        @inline function unsafe_map!(::Type{<:$engine},
                                     f::Function,
                                     dst::$array,
                                     x::$array)
            @vectorize $optim for i in eachindex(dst, x)
                dst[i] = f(x[i])
            end
            nothing
        end
        @inline function unsafe_map!(::Type{<:$engine},
                                     f::Function,
                                     dst::$array,
                                     x::$array,
                                     y::$array)
            @vectorize $optim for i in eachindex(dst, x, y)
                dst[i] = f(x[i], y[i])
            end
            nothing
        end
    end
end

# Generic implementations based on `map!`.
@inline function unsafe_map!(::Type{<:Engine},
                             f::Function,
                             dst::AbstractArray,
                             x::AbstractArray)
    map!(f, dst, x)
    nothing
end
@inline function unsafe_map!(::Type{<:Engine},
                             f::Function,
                             dst::AbstractArray,
                             x::AbstractArray,
                             y::AbstractArray)
    map!(f, dst, x, y)
    nothing
end
