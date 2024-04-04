"""
    NumOptBase.engine(args...) -> E::Type{<:NumOptBase.Engine}

yields the type of the implementation of numerical operations for array
arguments `args...`.

"""
engine() = Engine
@inline engine(::TurboArray...) = TurboLoopEngine
if SimdArray !== TurboArray
    @inline engine(::SimdArray...) = SimdLoopEngine
end
if AbstractArray !== SimdArray && AbstractArray !== TurboArray
    @inline engine(::AbstractArray...) = Engine
end

"""
    NumOptBase.@vectorize opt for ...

compiles the `for ...` loop according to optimization `opt`, one of:

- `:none` to perform bounds checking and no optimization;

- `:inbounds` to optimize with `@inbounds`;

- `:simd` to optimize with `@inbounds @simd`;

- `:avx` to optimize with `@inbounds @avx` (requires `LoopVectorization`);

- `:turbo` to optimize with `@inbounds @turbo` (requires `LoopVectorization`).

"""
macro vectorize(opt, expr)
    (expr isa Expr && expr.head === :for) || error("expecting a `for` loop after `@vectorize opt ...`")
    esc(_vectorize(opt, expr))
end

_vectorize(opt::QuoteNode, expr::Expr) = _vectorize(opt.value, expr)
_vectorize(opt::Any, expr::Expr) = error("`opt` must be a symbol in `@vectorize opt for ...`")
_vectorize(opt::Symbol, expr::Expr) =
    opt === :none     ?                     expr  :
    opt === :inbounds ? :(@inbounds        $expr) :
    opt === :simd     ? :(@inbounds @simd  $expr) :
    opt === :avx      ? :(@inbounds @avx   $expr) :
    opt === :turbo    ? :(@inbounds @turbo $expr) :
    error("unknown loop optimizer `:$opt`")

"""
    NumOptBase.check_axes(name, arr, ref)
    NumOptBase.check_axes(name, arr, rngs)

throws a `DimensionMismatch` exception if the axes of array `arr` are not
`rngs` or not the same as those of array `ref`. Argument `name` is used for the
error message.

"""
check_axes(name::AbstractString, arr::AbstractArray, ref::AbstractArray) =
    check_axes(name, arr, axes(ref))

function check_axes(name::AbstractString, arr::AbstractArray,
                    rngs::Tuple{Vararg{AbstractUnitRange{<:Integer}}})
    axes(arr) == rgns || throw(DimensionMismatch(pretty(
        name, " has incompatible axes, got ", axes(arr), " instead of ", rngs)))
    nothing
end

function check_axes(x::AbstractArray;
                    dest::Union{AbstractArray,Nothing} = nothing,
                    dir::Union{AbstractArray,Nothing} = nothing,
                    lower::Union{AbstractArray,Nothing} = nothing,
                    upper::Union{AbstractArray,Nothing} = nothing,)
    rngs = axes(x)
    dest  isa Nothing || check_axes("destination array", dest,    rngs)
    dir   isa Nothing || check_axes("search direction",  dir,     rngs)
    lower isa Nothing || check_axes("lower bound",       Ω.lower, rngs)
    upper isa Nothing || check_axes("upper bound",       Ω.upper, rngs)
    nothing
end

"""
    NumOptBase.pretty_print(io::IO, args...)

prints `args...` to `io` in a pretty form that is useful for error messages.
This method may be specialized on the type of each of `args...`. The default is
to call `print`.

"""
function pretty_print(io::IO, xs...)
    for x in xs
        pretty_print(io, x)
    end
    nothing
end
function pretty_print(io::IO, x::Tuple)
    n = length(x)
    print(io, '(')
    for i in 1:n
        pretty_print(io, x[i])
        (i == 1 || i < n) && print(io, ',')
    end
    print(io, ')')
    nothing
end
pretty_print(io::IO, x) = print(io::IO, x)
pretty_print(io::IO, rng::AbstractUnitRange) = print(io, first(rng), ':', last(rng))
pretty_print(io::IO, rng::OrdinalRange) = print(io, first(rng), ':', step(rng), ':', last(rng))

"""
    NumOptBase.pretty(args...) -> str

prints `args...` as a human readable string using [`NumOptBase.pretty_print`](@ref).

"""
function pretty(xs...)
    buf = IOBuffer()
    pretty_print(buf, xs...)
    return String(take!(buf))
end

"""
    NumOptBase.choice(t1::Bool, x1::T, t2::Bool, x2::T, ..., tn::Bool, xn::T, y::T)

yields `x1` if `t1` is true, otherwise `x2` if `t2` is true, and so on, and
finally `y` if none of the test values is true. This is equivalent to:

    t1 ? x1 :
    t2 ? x2 :
    ...
    tn ? xn : y

except that all expressions are computed. This is a generalization of the
`ifelse` function. The reason to use such a function rather than the `? :` syntax
is the possibility to eliminate branches in optimized code.

"""
@inline choice(t1::Bool, x1::T, y::T) where {T} =
    t1 ? x1 : y
@inline choice(t1::Bool, x1::T, t2::Bool, x2::T, y::T) where {T} =
    t1 ? x1 :
    t2 ? x2 : y
@inline choice(t1::Bool, x1::T, t2::Bool, x2::T, t3::Bool, x3::T, y::T) where {T} =
    t1 ? x1 :
    t2 ? x2 :
    t3 ? x3 : y

"""
    NumOptBase.only_arrays(args...) -> tup

yields a tuple of the arrays in `args...`.

"""
@inline only_arrays(A::Any) = ()
@inline only_arrays(A::AbstractArray...) = A # this includes empty tuple
@inline only_arrays(A::Any, B...) = only_arrays(B...)
@inline only_arrays(A::AbstractArray, B...) = (A, only_arrays(B...)...)

"""
    NumOptBase.flatten(x)

converts array `x` in to a vector.

"""
flatten(x::AbstractVector) = x
flatten(x::AbstractArray) = reshape(x, length(x))

"""
    NumOptBase.convert_multiplier(α, A)

converts scalar number `α` to a floating-point type whose numerical precision
is the same as that of the elements of abstract array `A`.

"""
convert_multiplier(α::Number, A::AbstractArray) = convert_floating_point_type(eltype(A), α)

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
@inline (f::αxpy)(x, y) = muladd(f.α, x, y) # f.α*x + y
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
                               (:simd,      SimdArray,     SimdLoopEngine))
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
        @inline function unsafe_map!(::Type{<:$engine},
                                     f::Function,
                                     dst::$array,
                                     x::$array,
                                     y::$array,
                                     z::$array)
            @vectorize $optim for i in eachindex(dst, x, y, z)
                dst[i] = f(x[i], y[i], z[i])
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
@inline function unsafe_map!(::Type{<:Engine},
                             f::Function,
                             dst::AbstractArray,
                             x::AbstractArray,
                             y::AbstractArray,
                             z::AbstractArray)
    map!(f, dst, x, y, z)
    nothing
end
