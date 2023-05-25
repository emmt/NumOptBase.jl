module BenchmarkingNumOptBase

export
    Debug, InBounds, Vectorize, Turbo,
    vdot, vnorm1, vnorm2, vnorminf,
    vmultiply!, vscale!, vupdate!, vcombine!

using BenchmarkTools
using ArrayTools: @assert_same_axes
using NumOptBase
using NumOptBase: convert_multiplier
using LoopVectorization
using MayOptimize
using Unitless

abstract type Turbo <: Vectorize end

macro vect(P, blk)
    # Type to compare `P` with is to be interpreted in this module context
    # and thus must not be escaped.  But `opt` and `blk` must be escaped.
    # I empirically found that `esc(:(@macro1 @macro2 ... $expr))` was the
    # correct way to escape expression `expr`, possibly preceded by some
    # other macro calls; the alternative `:(esc($expr))` works for all
    # expressions below but not for the one with `@simd` because this macro
    # expects a `for` loop.
    opt = esc(:($P))
    blk0 = esc(:($blk))
    blk1 = esc(:(@inbounds $blk))
    blk2 = esc(:(@inbounds @simd $blk))
    blk3 = esc(:(@turbo $blk))
    quote
        if $opt <: Turbo
            $blk3
        elseif $opt <: Vectorize
            $blk2
        elseif $opt <: InBounds
            $blk1
        elseif $opt <: Debug
            $blk0
        else
            Base.error("argument `P` of macro `@vect` is not a valid optimization level")
        end
    end
end

vdot(x::AbstractArray, y::AbstractArray) = NumOptBase.inner(x, y)
vdot(w::AbstractArray, x::AbstractArray, y::AbstractArray) = NumOptBase.inner(x, y)

vectorize(opt::Symbol, loop::Expr) =
    opt === :debug    ? loop :
    opt === :inbounds ? :(@inbounds $loop) :
    opt === :simd     ? :(@inbounds @simd $loop) :
    opt === :turbo    ? :(@turbo $loop) :
    error("unknown loop optimizer `$opt`")

macro vectorize(opt::Symbol, loop::Expr)
    esc(vectorize(opt, loop))
end

for opt in (:debug, :inbounds, :simd, :turbo)
    @eval begin
        function $(Symbol(:vdot_,opt))(x::AbstractArray{T,N},
                                       y::AbstractArray{T,N}) where {T<:Real,N}
            @assert_same_axes x y
            acc = zero(T)*zero(T)
            @vectorize $opt for i in eachindex(x, y)
                acc += x[i]*y[i]
            end
            return acc
        end
        function $(Symbol(:vdot_,opt))(w::AbstractArray{T,N},
                                       x::AbstractArray{T,N},
                                       y::AbstractArray{T,N}) where {T<:Real,N}
            @assert_same_axes w x y
            acc = zero(T)*zero(T)*zero(T)
            @vectorize $opt for i in eachindex(w, x, y)
                acc += w[i]*x[i]*y[i]
            end
            return acc
        end
    end
end

function vdot(::Type{L},
              x::AbstractArray{T,N},
              y::AbstractArray{T,N}) where {L<:OptimLevel,T<:Real,N}
    @assert_same_axes x y
    acc = zero(T)*zero(T)
    @vect L for i in eachindex(x, y)
        acc += x[i]*y[i]
    end
    return acc
end

function vdot(::Type{L},
              w::AbstractArray{T,N},
              x::AbstractArray{T,N},
              y::AbstractArray{T,N}) where {L<:OptimLevel,T<:Real,N}
    @assert_same_axes w x y
    acc = zero(T)*zero(T)*zero(T)
    @vect L for i in eachindex(w, x, y)
        acc += w[i]*x[i]*y[i]
    end
    return acc
end

vnorm1(x::AbstractArray) =  NumOptBase.norm1(x)
vnorm2(x::AbstractArray) =  NumOptBase.norm2(x)
vnorminf(x::AbstractArray) =  NumOptBase.norminf(x)

for opt in (:debug, :inbounds, :simd, :turbo)
    @eval begin

        function $(Symbol(:vnorm1_,opt))(x::AbstractArray{T,N}) where {T<:Real,N}
            acc = abs(zero(T))
            @vectorize $opt for i in eachindex(x)
                acc += abs(x[i])
            end
            return acc
        end

        function $(Symbol(:vnorm2_,opt))(x::AbstractArray{T,N}) where {T<:Real,N}
            acc = abs2(zero(T))
            @vectorize $opt for i in eachindex(x)
                acc += abs2(x[i])
            end
            return sqrt(acc)
        end

        function $(Symbol(:vnorminf_,opt))(x::AbstractArray{T,N}) where {T<:Real,N}
            r = abs(zero(T))
            @vectorize $opt for i in eachindex(x)
                a = abs(x[i])
                r = a > r ? a : r
            end
            return r
        end

    end
end

function vnorm1(::Type{L},
                x::AbstractArray{T,N}) where {L<:OptimLevel,T<:Real,N}
    acc = abs(zero(T))
    @vect L for i in eachindex(x)
        acc += abs(x[i])
    end
    return acc
end

function vnorm2(::Type{L},
                x::AbstractArray{T,N}) where {L<:OptimLevel,T<:Real,N}
    acc = abs2(zero(T))
    @vect L for i in eachindex(x)
        acc += abs2(x[i])
    end
    return sqrt(acc)
end

function vnorminf(::Type{L},
                  x::AbstractArray{T,N}) where {L<:OptimLevel,T<:Real,N}
    r = abs(zero(T))
    @vect L for i in eachindex(x)
        a = abs(x[i])
        r = a > r ? a : r
    end
    return r
end

vscale!(dst::AbstractArray, α::Real, x::AbstractArray) = NumOptBase.scale!(dst, α, x)

function vscale!(::Type{L},
                 dst::AbstractArray{T,N},
                 α::Real, x::AbstractArray{T,N}) where {L<:OptimLevel,T,N}
    @assert_same_axes dst x
    α = convert_multiplier(α, x)
    @vect L for i in eachindex(dst, x)
        dst[i] = α*x[i]
    end
    return dst
end

vupdate!(dst::AbstractArray, α::Real, x::AbstractArray) = NumOptBase.update!(dst, α, x)

function vupdate!(::Type{L},
                  dst::AbstractArray{T,N},
                  α::Real, x::AbstractArray{T,N}) where {L<:OptimLevel,T,N}
    @assert_same_axes dst x
    α = convert_multiplier(α, x)
    @vect L for i in eachindex(dst, x)
        dst[i] += α*x[i]
    end
    return dst
end

vmultiply!(dst::AbstractArray, x::AbstractArray, y::AbstractArray) = NumOptBase.multiply!(dst, x, y)

function vmultiply!(::Type{L},
                    dst::AbstractArray{T,N},
                    x::AbstractArray{T,N},
                    y::AbstractArray{T,N}) where {L<:OptimLevel,T,N}
    @assert_same_axes dst x y
    @vect L for i in eachindex(dst, x, y)
        dst[i] = x[i]*y[i]
    end
    return dst
end

vcombine!(dst::AbstractArray, α::Real, x::AbstractArray, β::Real, y::AbstractArray) =
    NumOptBase.combine!(dst, α, x, β, y)

function vcombine!(::Type{L},
                    dst::AbstractArray{T,N},
                    α::Real, x::AbstractArray{T,N},
                    β::Real, y::AbstractArray{T,N}) where {L<:OptimLevel,T,N}
    @assert_same_axes dst x y
    α = convert_multiplier(α, x)
    β = convert_multiplier(β, y)
    @vect L for i in eachindex(dst, x, y)
        dst[i] = α*x[i] + β*y[i]
    end
    return dst
end

for opt in (:debug, :inbounds, :simd, :turbo)

    @eval function $(Symbol(:vscale_,opt,:(!)))(dst::AbstractArray{T,N},
                                                α::Real, x::AbstractArray{T,N}) where {T,N}
        @assert_same_axes dst x
        α = convert_multiplier(α, x)
        @vectorize $opt for i in eachindex(dst, x)
            dst[i] = α*x[i]
        end
        return dst
    end

    @eval function $(Symbol(:vupdate_,opt,:(!)))(dst::AbstractArray{T,N},
                                                 α::Real, x::AbstractArray{T,N}) where {T,N}
        @assert_same_axes dst x
        α = convert_multiplier(α, x)
        @vectorize $opt for i in eachindex(dst, x)
            dst[i] += α*x[i]
        end
        return dst
    end

    @eval function $(Symbol(:vmultiply_,opt,:(!)))(dst::AbstractArray{T,N},
                                                   x::AbstractArray{T,N},
                                                   y::AbstractArray{T,N}) where {T,N}
        @assert_same_axes dst x y
        @vectorize $opt for i in eachindex(dst, x, y)
            dst[i] = x[i]*y[i]
        end
        return dst
    end

    @eval function $(Symbol(:vcombine_,opt,:(!)))(dst::AbstractArray{T,N},
                                                  α::Real, x::AbstractArray{T,N},
                                                  β::Real, y::AbstractArray{T,N}) where {T,N}
        @assert_same_axes dst x y
        α = convert_multiplier(α, x)
        β = convert_multiplier(β, y)
        @vectorize $opt for i in eachindex(dst, x, y)
            dst[i] = α*x[i] + β*y[i]
        end
        return dst
    end

end

runtests(; T::Type=Float32, dims::Dims=(10_000,), kwds...) =
    runtests(rand(T, dims), rand(T, dims), rand(T, dims); kwds...)

function runtests(w::AbstractArray, x::AbstractArray, y::AbstractArray;
                  α::Real = 1.2, β::Real = -3.7,
                  ops::Union{Symbol,Tuple{Vararg{Symbol}}} = :all)
    if ops isa Symbol
        ops = (ops,)
    end
    z = similar(x)
    if :all ∈ ops || :inner ∈ ops
        println("Inner product")
        print("  NumOptBase.inner(  x,y) "); @btime NumOptBase.inner(    $x, $y)
        print("  vdot_debug(        x,y) "); @btime vdot_debug(          $x, $y)
        print("  vdot_inbounds(     x,y) "); @btime vdot_inbounds(       $x, $y)
        print("  vdot_simd(         x,y) "); @btime vdot_simd(           $x, $y)
        print("  vdot_turbo(        x,y) "); @btime vdot_turbo(          $x, $y)
        print("  vdot(InBounds,     x,y) "); @btime vdot($InBounds,      $x, $y)
        print("  vdot(Vectorize,    x,y) "); @btime vdot($Vectorize,     $x, $y)
        print("  vdot(Turbo,        x,y) "); @btime vdot($Turbo,         $x, $y)
        print("  NumOptBase.inner(w,x,y) "); @btime NumOptBase.inner($w, $x, $y)
        print("  vdot_debug(      w,x,y) "); @btime vdot_debug(      $w, $x, $y)
        print("  vdot_inbounds(   w,x,y) "); @btime vdot_inbounds(   $w, $x, $y)
        print("  vdot_simd(       w,x,y) "); @btime vdot_simd(       $w, $x, $y)
        print("  vdot_turbo(      w,x,y) "); @btime vdot_turbo(      $w, $x, $y)
        print("  vdot(InBounds,   w,x,y) "); @btime vdot($InBounds,  $w, $x, $y)
        print("  vdot(Vectorize,  w,x,y) "); @btime vdot($Vectorize, $w, $x, $y)
        print("  vdot(Turbo,      w,x,y) "); @btime vdot($Turbo,     $w, $x, $y)
    end
    if :all ∈ ops || :norm ∈ ops || :norm1 ∈ ops
        println()
        println("ℓ₁ norm")
        print("  NumOptBase.norm1( x) "); @btime NumOptBase.norm1(  $x)
        print("  vnorm1_debug(     x) "); @btime vnorm1_debug(      $x)
        print("  vnorm1_inbounds(  x) "); @btime vnorm1_inbounds(   $x)
        print("  vnorm1_simd(      x) "); @btime vnorm1_simd(       $x)
        print("  vnorm1_turbo(     x) "); @btime vnorm1_turbo(      $x)
        print("  vnorm1(InBounds,  x) "); @btime vnorm1($InBounds,  $x)
        print("  vnorm1(Vectorize, x) "); @btime vnorm1($Vectorize, $x)
        print("  vnorm1(Turbo,     x) "); @btime vnorm1($Turbo,     $x)
    end
    if :all ∈ ops || :norm ∈ ops || :norm2 ∈ ops
        println()
        println("Euclidean norm")
        print("  NumOptBase.norm2( x) "); @btime NumOptBase.norm2(  $x)
        print("  vnorm2_debug(     x) "); @btime vnorm2_debug(      $x)
        print("  vnorm2_inbounds(  x) "); @btime vnorm2_inbounds(   $x)
        print("  vnorm2_simd(      x) "); @btime vnorm2_simd(       $x)
        print("  vnorm2_turbo(     x) "); @btime vnorm2_turbo(      $x)
        print("  vnorm2(InBounds,  x) "); @btime vnorm2($InBounds,  $x)
        print("  vnorm2(Vectorize, x) "); @btime vnorm2($Vectorize, $x)
        print("  vnorm2(Turbo,     x) "); @btime vnorm2($Turbo,     $x)
    end
    if :all ∈ ops || :norm ∈ ops || :norminf ∈ ops
        println()
        println("Infinite norm")
        print("  NumOptBase.norminf( x) "); @btime NumOptBase.norminf(  $x)
        print("  vnorminf_debug(     x) "); @btime vnorminf_debug(      $x)
        print("  vnorminf_inbounds(  x) "); @btime vnorminf_inbounds(   $x)
        print("  vnorminf_simd(      x) "); @btime vnorminf_simd(       $x)
        print("  vnorminf_turbo(     x) "); @btime vnorminf_turbo(      $x)
        print("  vnorminf(InBounds,  x) "); @btime vnorminf($InBounds,  $x)
        print("  vnorminf(Vectorize, x) "); @btime vnorminf($Vectorize, $x)
        print("  vnorminf(Turbo,     x) "); @btime vnorminf($Turbo,     $x)
    end
    if :all ∈ ops || :scale ∈ ops
        println()
        println("Scaling")
        print("  NumOptBase.scale!( z, α, x) "); @btime NumOptBase.scale!(  $z, $α, $x)
        print("  vscale_debug!(     z, α, x) "); @btime vscale_debug!(      $z, $α, $x)
        print("  vscale_inbounds!(  z, α, x) "); @btime vscale_inbounds!(   $z, $α, $x)
        print("  vscale_simd!(      z, α, x) "); @btime vscale_simd!(       $z, $α, $x)
        print("  vscale_turbo!(     z, α, x) "); @btime vscale_turbo!(      $z, $α, $x)
        print("  vscale!(InBounds,  z, α, x) "); @btime vscale!($InBounds,  $z, $α, $x)
        print("  vscale!(Vectorize, z, α, x) "); @btime vscale!($Vectorize, $z, $α, $x)
        print("  vscale!(Turbo,     z, α, x) "); @btime vscale!($Turbo,     $z, $α, $x)
    end
    if :all ∈ ops || :update ∈ ops
        println()
        println("Updating")
        print("  NumOptBase.update!( z, α, x) "); @btime NumOptBase.update!(  $z, $α, $x)
        print("  vupdate_debug!(     z, α, x) "); @btime vupdate_debug!(      $z, $α, $x)
        print("  vupdate_inbounds!(  z, α, x) "); @btime vupdate_inbounds!(   $z, $α, $x)
        print("  vupdate_simd!(      z, α, x) "); @btime vupdate_simd!(       $z, $α, $x)
        print("  vupdate_turbo!(     z, α, x) "); @btime vupdate_turbo!(      $z, $α, $x)
        print("  vupdate!(InBounds,  z, α, x) "); @btime vupdate!($InBounds,  $z, $α, $x)
        print("  vupdate!(Vectorize, z, α, x) "); @btime vupdate!($Vectorize, $z, $α, $x)
        print("  vupdate!(Turbo,     z, α, x) "); @btime vupdate!($Turbo,     $z, $α, $x)
    end
    if :all ∈ ops || :multiply ∈ ops
        println()
        println("Multiplying")
        print("  NumOptBase.multiply!( z, x, y) "); @btime NumOptBase.multiply!(  $z, $x, $y)
        print("  vmultiply_debug!(     z, x, y) "); @btime vmultiply_debug!(      $z, $x, $y)
        print("  vmultiply_inbounds!(  z, x, y) "); @btime vmultiply_inbounds!(   $z, $x, $y)
        print("  vmultiply_simd!(      z, x, y) "); @btime vmultiply_simd!(       $z, $x, $y)
        print("  vmultiply_turbo!(     z, x, y) "); @btime vmultiply_turbo!(      $z, $x, $y)
        print("  vmultiply!(InBounds,  z, x, y) "); @btime vmultiply!($InBounds,  $z, $x, $y)
        print("  vmultiply!(Vectorize, z, x, y) "); @btime vmultiply!($Vectorize, $z, $x, $y)
        print("  vmultiply!(Turbo,     z, x, y) "); @btime vmultiply!($Turbo,     $z, $x, $y)
    end
    if :all ∈ ops || :combine ∈ ops
        println()
        println("Combining")
        print("  NumOptBase.combine!( z, α, x, β, y) "); @btime NumOptBase.combine!(  $z, $α, $x, $β, $y)
        print("  vcombine_debug!(     z, α, x, β, y) "); @btime vcombine_debug!(      $z, $α, $x, $β, $y)
        print("  vcombine_inbounds!(  z, α, x, β, y) "); @btime vcombine_inbounds!(   $z, $α, $x, $β, $y)
        print("  vcombine_simd!(      z, α, x, β, y) "); @btime vcombine_simd!(       $z, $α, $x, $β, $y)
        print("  vcombine_turbo!(     z, α, x, β, y) "); @btime vcombine_turbo!(      $z, $α, $x, $β, $y)
        print("  vcombine!(InBounds,  z, α, x, β, y) "); @btime vcombine!($InBounds,  $z, $α, $x, $β, $y)
        print("  vcombine!(Vectorize, z, α, x, β, y) "); @btime vcombine!($Vectorize, $z, $α, $x, $β, $y)
        print("  vcombine!(Turbo,     z, α, x, β, y) "); @btime vcombine!($Turbo,     $z, $α, $x, $β, $y)
    end
    nothing
end

end # module
