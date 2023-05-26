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

# Replace base code of some NumOptBase operations for strided arrays.

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

end # module
