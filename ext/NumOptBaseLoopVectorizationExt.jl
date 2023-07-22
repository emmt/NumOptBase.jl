module NumOptBaseLoopVectorizationExt

if isdefined(Base, :get_extension)
    using LoopVectorization
    import NumOptBase
    using NumOptBase: inner, norm1, norminf
else
    using ..LoopVectorization
    import ..NumOptBase
    using ..NumOptBase: inner, norm1, norminf
end

# The @turbo macro was introduced in LoopVectorization 0.12.22 to replace @avx.
# We could either always use the @turbo macro in the code and, if it does not
# exist, define an alias with:
#
#    const var"@turbo" = var"@avx"
#
# or do the opposite (always use @avx in the code and make it an alias to
# @turbo if it does not exist). Since the var"..." macro requires Julia ≥ 1.3
# and since LoopVectorization 0.12.22 requires Julia ≥ 1.5, we should take the
# latter solution.
#
# Nevertheless, it turns out that LoopVectorization yields broken code on Julia
# < 1.5 (the result is as if the loop is not executed, may be due to
# LoopVectorization ~ 0.8), so we just skip improved loop vectorization if the
# @turbo macro is absent.

@static if isdefined(@__MODULE__, Symbol("@turbo"))

    using Base: HWReal

    const can_avx = isdefined(LoopVectorization, :ArrayInterface) &&
        isdefined(LoopVectorization.ArrayInterface, :can_avx) ?
        LoopVectorization.ArrayInterface.can_avx : f -> false

    function NumOptBase.unsafe_inner(x::StridedArray{T,N},
                                     y::StridedArray{T,N}) where {T<:HWReal,N}
        acc = inner(zero(eltype(x)), zero(eltype(y)))
        @turbo for i in eachindex(x, y)
            acc += inner(x[i], y[i])
        end
        return acc
    end

    function NumOptBase.unsafe_inner(w::StridedArray{T,N},
                                     x::StridedArray{T,N},
                                     y::StridedArray{T,N}) where {T<:HWReal,N}
        acc = inner(zero(eltype(w)), zero(eltype(x)), zero(eltype(y)))
        @turbo for i in eachindex(w, x, y)
            acc += inner(w[i], x[i], y[i])
        end
        return acc
    end

    function NumOptBase.norm1(x::StridedArray{T,N}) where {T<:HWReal,N}
        acc = zero(eltype(x))
        @turbo for i in eachindex(x)
            acc += abs(x[i])
        end
        return acc
    end

    function NumOptBase.norm2(x::StridedArray{T,N}) where {T<:HWReal,N}
        acc = zero(eltype(x))
        @turbo for i in eachindex(x)
            acc += abs2(x[i])
        end
        return sqrt(acc)
    end

    function NumOptBase.norminf(x::StridedArray{T,N}) where {T<:HWReal,N}
        r = zero(eltype(x))
        @turbo for i in eachindex(x)
            a = abs(x[i])
            r = a > r ? a : r
        end
        return r
    end

    @inline function NumOptBase.unsafe_map!(f::Function,
                                            dst::StridedArray{T,N},
                                            x::StridedArray{T,N}) where {T<:HWReal,N}
        if can_avx(f)
            @turbo for i in eachindex(dst, x)
                dst[i] = f(x[i])
            end
        else
            @inbounds @simd for i in eachindex(dst, x)
                dst[i] = f(x[i])
            end
        end
        nothing
    end

    @inline function NumOptBase.unsafe_map!(f::Function,
                                            dst::StridedArray{T,N},
                                            x::StridedArray{T,N},
                                            y::StridedArray{T,N}) where {T<:HWReal,N}
        if can_avx(f)
            @turbo for i in eachindex(dst, x, y)
                dst[i] = f(x[i], y[i])
            end
        else
            @inbounds @simd for i in eachindex(dst, x, y)
                dst[i] = f(x[i], y[i])
            end
        end
        nothing
    end

else

    @warn """We are currently skipping improved optimizations by `LoopVectorization`
             because the `@turbo` macro is absent. This maybe due to your Julia
             version ($VERSION) being too old or to an outdated version of
             the `LoopVectorization` package. Please consider upgrading."""

end

end # module
