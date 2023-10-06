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
    NumOptBase.CudaEngine <: NumOptBase.Engine

is the abstract type identifying implementation for CUDA arrays.

"""
abstract type CudaEngine <: Engine end

abstract type MapEngine <: Engine end

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

"""
    NumOptBase.unsafe_inner!(E, [w,] x, y)

executes the task of [`NumOptBase.inner!`](@ref) assuming without checking that
array arguments have the same axes. This method is thus *unsafe* and shall not
be directly called but it may be extended for specific array types. By default,
it uses SIMD vectorization for strided arrays and calls `mapreduce` for other
arrays. Argument `E` specifies which *engine* to be use for the computations.

""" unsafe_inner!

# Loop-based implementations.
for (optim, array, engine) in ((:none,      AbstractArray, LoopEngine),
                               (:inbounds,  AbstractArray, InBoundsLoopEngine),
                               (:simd,      StridedArray,  SimdLoopEngine))
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
