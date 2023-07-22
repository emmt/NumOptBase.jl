module BenchmarkingNumOptBase

export
    Debug, InBounds, Vectorize, Turbo,
    vdot, vnorm1, vnorm2, vnorminf,
    vmultiply!, vscale!, vupdate!, vcombine!

using BenchmarkTools
using ArrayTools: @assert_same_axes
using NumOptBase
using NumOptBase: convert_multiplier, @vectorize
using LoopVectorization
using MayOptimize
using Unitless

const use_avx   = isdefined(@__MODULE__, Symbol("@avx"))
const use_turbo = isdefined(@__MODULE__, Symbol("@turbo"))

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
    if use_turbo || use_avx
        if use_turbo
            blk3 = esc(:(@turbo $blk))
        else
            blk3 = esc(:(@avx $blk))
        end
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
    else
        quote
            if $opt <: Vectorize
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
end

vdot(x::AbstractArray, y::AbstractArray) = NumOptBase.inner(x, y)
vdot(w::AbstractArray, x::AbstractArray, y::AbstractArray) = NumOptBase.inner(x, y)

for opt in (:none, :inbounds, :simd, :turbo)
    if opt === :turbo && ! use_turbo
        use_avx || continue
        opt = :avx
    end
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

for opt in (:none, :inbounds, :simd, :turbo)
    if opt === :turbo && ! use_turbo
        use_avx || continue
        opt = :avx
    end
    (opt !== :turbo) || use_turbo || use_avx || continue
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

for opt in (:none, :inbounds, :simd, :turbo)
    if opt === :turbo && ! use_turbo
        use_avx || continue
        opt = :avx
    end
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

function copy_memcpy!(dst::AbstractArray{T,N}, src::AbstractArray{T,N}) where {T,N}
    @assert isbitstype(T)
    @assert_same_axes dst src
    nbytes = length(dst)*sizeof(T)
    ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), dst, src, nbytes)
    return dst
end
function copy_inbounds!(dst::AbstractArray{T,N}, src::AbstractArray{T,N}) where {T,N}
    @assert_same_axes dst src
    @inbounds for i in eachindex(dst, src)
        dst[i] = src[i]
    end
    return dst
end
function copy_simd!(dst::AbstractArray{T,N}, src::AbstractArray{T,N}) where {T,N}
    @assert_same_axes dst src
    @inbounds @simd for i in eachindex(dst, src)
        dst[i] = src[i]
    end
    return dst
end
@static if use_turbo
    function copy_turbo!(dst::AbstractArray{T,N}, src::AbstractArray{T,N}) where {T,N}
        @assert_same_axes dst src
        @turbo for i in eachindex(dst, src)
            dst[i] = src[i]
        end
        return dst
    end
elseif use_avx
    function copy_turbo!(dst::AbstractArray{T,N}, src::AbstractArray{T,N}) where {T,N}
        @assert_same_axes dst src
        @avx for i in eachindex(dst, src)
            dst[i] = src[i]
        end
        return dst
    end
end

zerofill_fill!(A::AbstractArray{T}) where {T} = fill!(A, zero(T))
function zerofill_memset!(A::AbstractArray{T}) where {T}
    @assert isbitstype(T)
    nbytes = length(A)*sizeof(T)
    ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), A, 0, nbytes)
    return A
end
function zerofill_inbounds!(A::AbstractArray{T}) where {T}
    val = zero(T)
    @inbounds for i in eachindex(A)
        A[i] = val
    end
    return A
end
function zerofill_simd!(A::AbstractArray{T}) where {T}
    val = zero(T)
    @inbounds @simd for i in eachindex(A)
        A[i] = val
    end
    return A
end
@static if use_turbo
    function zerofill_turbo!(A::AbstractArray{T}) where {T}
        val = zero(T)
        @turbo for i in eachindex(A)
            A[i] = val
        end
        return A
    end
elseif use_avx
    function zerofill_turbo!(A::AbstractArray{T}) where {T}
        val = zero(T)
        @avx for i in eachindex(A)
            A[i] = val
        end
        return A
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
    n = length(x)
    z = similar(x)
    if :all ∈ ops || :inner ∈ ops
        println("Inner product (≈ $(2n) or $(3n) ops)")
        print(    "  NumOptBase.inner(  x,y) "); @btime NumOptBase.inner(    $x, $y)
        print(    "  vdot_none(         x,y) "); @btime vdot_none(           $x, $y)
        print(    "  vdot_inbounds(     x,y) "); @btime vdot_inbounds(       $x, $y)
        print(    "  vdot_simd(         x,y) "); @btime vdot_simd(           $x, $y)
        if use_turbo || use_avx
            print("  vdot_turbo(        x,y) "); @btime vdot_turbo(          $x, $y)
        end
        print(    "  vdot(InBounds,     x,y) "); @btime vdot($InBounds,      $x, $y)
        print(    "  vdot(Vectorize,    x,y) "); @btime vdot($Vectorize,     $x, $y)
        if use_turbo || use_avx
            print("  vdot(Turbo,        x,y) "); @btime vdot($Turbo,         $x, $y)
        end
        print(    "  NumOptBase.inner(w,x,y) "); @btime NumOptBase.inner($w, $x, $y)
        print(    "  vdot_none(       w,x,y) "); @btime vdot_none(       $w, $x, $y)
        print(    "  vdot_inbounds(   w,x,y) "); @btime vdot_inbounds(   $w, $x, $y)
        print(    "  vdot_simd(       w,x,y) "); @btime vdot_simd(       $w, $x, $y)
        if use_turbo || use_avx
            print("  vdot_turbo(      w,x,y) "); @btime vdot_turbo(      $w, $x, $y)
        end
        print(    "  vdot(InBounds,   w,x,y) "); @btime vdot($InBounds,  $w, $x, $y)
        print(    "  vdot(Vectorize,  w,x,y) "); @btime vdot($Vectorize, $w, $x, $y)
        if use_turbo || use_avx
            print("  vdot(Turbo,      w,x,y) "); @btime vdot($Turbo,     $w, $x, $y)
        end
    end
    if :all ∈ ops || :norm ∈ ops || :norm1 ∈ ops
        println()
        println("ℓ₁ norm (≈ $(2n) ops)")
        print(    "  NumOptBase.norm1( x) "); @btime NumOptBase.norm1(  $x)
        print(    "  vnorm1_none(     x) "); @btime vnorm1_none(      $x)
        print(    "  vnorm1_inbounds(  x) "); @btime vnorm1_inbounds(   $x)
        print(    "  vnorm1_simd(      x) "); @btime vnorm1_simd(       $x)
        if use_turbo || use_avx
            print("  vnorm1_turbo(     x) "); @btime vnorm1_turbo(      $x)
        end
        print(    "  vnorm1(InBounds,  x) "); @btime vnorm1($InBounds,  $x)
        print(    "  vnorm1(Vectorize, x) "); @btime vnorm1($Vectorize, $x)
        if use_turbo || use_avx
            print("  vnorm1(Turbo,     x) "); @btime vnorm1($Turbo,     $x)
        end
    end
    if :all ∈ ops || :norm ∈ ops || :norm2 ∈ ops
        println()
        println("Euclidean norm (≈ $(2n) ops)")
        print(    "  NumOptBase.norm2( x) "); @btime NumOptBase.norm2(  $x)
        print(    "  vnorm2_none(      x) "); @btime vnorm2_none(       $x)
        print(    "  vnorm2_inbounds(  x) "); @btime vnorm2_inbounds(   $x)
        print(    "  vnorm2_simd(      x) "); @btime vnorm2_simd(       $x)
        if use_turbo || use_avx
            print("  vnorm2_turbo(     x) "); @btime vnorm2_turbo(      $x)
        end
        print(    "  vnorm2(InBounds,  x) "); @btime vnorm2($InBounds,  $x)
        print(    "  vnorm2(Vectorize, x) "); @btime vnorm2($Vectorize, $x)
        if use_turbo || use_avx
            print("  vnorm2(Turbo,     x) "); @btime vnorm2($Turbo,     $x)
        end
    end
    if :all ∈ ops || :norm ∈ ops || :norminf ∈ ops
        println()
        println("Infinite norm (≈ $(2n) ops)")
        print(    "  NumOptBase.norminf( x) "); @btime NumOptBase.norminf(  $x)
        print(    "  vnorminf_none(      x) "); @btime vnorminf_none(       $x)
        print(    "  vnorminf_inbounds(  x) "); @btime vnorminf_inbounds(   $x)
        print(    "  vnorminf_simd(      x) "); @btime vnorminf_simd(       $x)
        if use_turbo || use_avx
            print("  vnorminf_turbo(     x) "); @btime vnorminf_turbo(      $x)
        end
        print(    "  vnorminf(InBounds,  x) "); @btime vnorminf($InBounds,  $x)
        print(    "  vnorminf(Vectorize, x) "); @btime vnorminf($Vectorize, $x)
        if use_turbo || use_avx
            print("  vnorminf(Turbo,     x) "); @btime vnorminf($Turbo,     $x)
        end
    end
    if :all ∈ ops || :zerofill ∈ ops
        println()
        println("Zero-filling (≈ $(n) ops)")
        print(    "  NumOptBase.zerofill!( z) "); @btime NumOptBase.zerofill!($z)
        print(    "  zerofill_fill!(       z) "); @btime zerofill_fill!(      $z)
        print(    "  zerofill_memset!(     z) "); @btime zerofill_memset!(    $z)
        print(    "  zerofill_inbounds!(   z) "); @btime zerofill_inbounds!(  $z)
        print(    "  zerofill_simd!(       z) "); @btime zerofill_simd!(      $z)
        if use_turbo || use_avx
            print("  zerofill_turbo!(      z) "); @btime zerofill_turbo!(     $z)
        end
    end
    if :all ∈ ops || :copy ∈ ops
        println()
        println("Copying (≈ $(n) ops)")
        print(    "  NumOptBase.copy!(z, x) "); @btime NumOptBase.copy!($z, $x)
        print(    "  copyto!(         z, x) "); @btime copyto!(         $z, $x)
        print(    "  copy_memcpy!(    z, x) "); @btime copy_memcpy!(    $z, $x)
        print(    "  copy_inbounds!(  z, x) "); @btime copy_inbounds!(  $z, $x)
        print(    "  copy_simd!(      z, x) "); @btime copy_simd!(      $z, $x)
        if use_turbo || use_avx
            print("  copy_turbo!(     z, x) "); @btime copy_turbo!(     $z, $x)
        end
    end
    if :all ∈ ops || :scale ∈ ops
        println()
        println("Scaling (≈ $(n) ops)")
        print(    "  NumOptBase.scale!( z, α, x) "); @btime NumOptBase.scale!(  $z, $α, $x)
        print(    "  vscale_none!(     z, α, x) "); @btime vscale_none!(      $z, $α, $x)
        print(    "  vscale_inbounds!(  z, α, x) "); @btime vscale_inbounds!(   $z, $α, $x)
        print(    "  vscale_simd!(      z, α, x) "); @btime vscale_simd!(       $z, $α, $x)
        if use_turbo || use_avx
            print("  vscale_turbo!(     z, α, x) "); @btime vscale_turbo!(      $z, $α, $x)
        end
        print(    "  vscale!(InBounds,  z, α, x) "); @btime vscale!($InBounds,  $z, $α, $x)
        print(    "  vscale!(Vectorize, z, α, x) "); @btime vscale!($Vectorize, $z, $α, $x)
        if use_turbo || use_avx
            print("  vscale!(Turbo,     z, α, x) "); @btime vscale!($Turbo,     $z, $α, $x)
        elseif use_avx
        end
    end
    if :all ∈ ops || :update ∈ ops
        println()
        println("Updating (≈ $(2n) ops)")
        print(    "  NumOptBase.update!( z, α, x) "); @btime NumOptBase.update!(  $z, $α, $x)
        print(    "  vupdate_none!(      z, α, x) "); @btime vupdate_none!(       $z, $α, $x)
        print(    "  vupdate_inbounds!(  z, α, x) "); @btime vupdate_inbounds!(   $z, $α, $x)
        print(    "  vupdate_simd!(      z, α, x) "); @btime vupdate_simd!(       $z, $α, $x)
        if use_turbo || use_avx
            print("  vupdate_turbo!(     z, α, x) "); @btime vupdate_turbo!(      $z, $α, $x)
        end
        print(    "  vupdate!(InBounds,  z, α, x) "); @btime vupdate!($InBounds,  $z, $α, $x)
        print(    "  vupdate!(Vectorize, z, α, x) "); @btime vupdate!($Vectorize, $z, $α, $x)
        if use_turbo || use_avx
            print("  vupdate!(Turbo,     z, α, x) "); @btime vupdate!($Turbo,     $z, $α, $x)
        end
    end
    if :all ∈ ops || :multiply ∈ ops
        println()
        println("Multiplying (≈ $(n) ops)")
        print(    "  NumOptBase.multiply!( z, x, y) "); @btime NumOptBase.multiply!(  $z, $x, $y)
        print(    "  vmultiply_none!(      z, x, y) "); @btime vmultiply_none!(       $z, $x, $y)
        print(    "  vmultiply_inbounds!(  z, x, y) "); @btime vmultiply_inbounds!(   $z, $x, $y)
        print(    "  vmultiply_simd!(      z, x, y) "); @btime vmultiply_simd!(       $z, $x, $y)
        if use_turbo || use_avx
            print("  vmultiply_turbo!(     z, x, y) "); @btime vmultiply_turbo!(      $z, $x, $y)
        end
        print(    "  vmultiply!(InBounds,  z, x, y) "); @btime vmultiply!($InBounds,  $z, $x, $y)
        print(    "  vmultiply!(Vectorize, z, x, y) "); @btime vmultiply!($Vectorize, $z, $x, $y)
        if use_turbo || use_avx
            print("  vmultiply!(Turbo,     z, x, y) "); @btime vmultiply!($Turbo,     $z, $x, $y)
        end
    end
    if :all ∈ ops || :combine ∈ ops
        println()
        println("Combining (≈ $(3n) ops)")
        print(    "  NumOptBase.combine!( z, α, x, β, y) "); @btime NumOptBase.combine!(  $z, $α, $x, $β, $y)
        print(    "  vcombine_none!(      z, α, x, β, y) "); @btime vcombine_none!(       $z, $α, $x, $β, $y)
        print(    "  vcombine_inbounds!(  z, α, x, β, y) "); @btime vcombine_inbounds!(   $z, $α, $x, $β, $y)
        print(    "  vcombine_simd!(      z, α, x, β, y) "); @btime vcombine_simd!(       $z, $α, $x, $β, $y)
        if use_turbo || use_avx
            print("  vcombine_turbo!(     z, α, x, β, y) "); @btime vcombine_turbo!(      $z, $α, $x, $β, $y)
        end
        print(    "  vcombine!(InBounds,  z, α, x, β, y) "); @btime vcombine!($InBounds,  $z, $α, $x, $β, $y)
        print(    "  vcombine!(Vectorize, z, α, x, β, y) "); @btime vcombine!($Vectorize, $z, $α, $x, $β, $y)
        if use_turbo || use_avx
            print("  vcombine!(Turbo,     z, α, x, β, y) "); @btime vcombine!($Turbo,     $z, $α, $x, $β, $y)
        end
    end
    nothing
end

end # module
