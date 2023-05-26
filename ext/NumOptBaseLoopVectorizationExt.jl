module NumOptBaseLoopVectorizationExt

if isdefined(Base, :get_extension)
    using LoopVectorization
    import NumOptBase
    using NumOptBase: inner
else
    using ..LoopVectorization
    import ..NumOptBase
    using ..NumOptBase: inner
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

    function NumOptBase.unsafe_inner(x::StridedArray,
                                     y::StridedArray)
        acc = inner(zero(eltype(x)), zero(eltype(y)))
        @turbo for i in eachindex(x, y)
            acc += inner(x[i], y[i])
        end
        return acc
    end

    function NumOptBase.unsafe_inner(w::StridedArray,
                                     x::StridedArray,
                                     y::StridedArray)
        acc = inner(zero(eltype(w)), zero(eltype(x)), zero(eltype(y)))
        @turbo for i in eachindex(w, x, y)
            acc += inner(w[i], x[i], y[i])
        end
        return acc
    end

    function NumOptBase.norminf(x::StridedArray)
        r = abs(zero(eltype(x)))
        @turbo for i in eachindex(x)
            a = abs(x[i])
            r = a > r ? a : r
        end
        return r
    end

else

    @warn """We are currently skipping improved optimizations by `LoopVectorization`
        because the `@turbo` macro is absent. This maybe due to your Julia version ($VERSION)
        being too old or to an outdated version of the `LoopVectorization` package.
        Please consider upgrading."""

end

end # module