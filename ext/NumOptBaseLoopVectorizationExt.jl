module NumOptBaseLoopVectorizationExt

if isdefined(Base, :get_extension)
    using LoopVectorization
    using NumOptBase
    using NumOptBase:
        PlusOrMinus,
        SimdLoopEngine,
        TurboArray,
        TurboLoopEngine,
        get_bound,
        can_vary,
        only_arrays,
        project_direction,
        project_variable
    import NumOptBase:
        norm1,
        norm2,
        norminf,
        unsafe_inner,
        unsafe_map!,
        unsafe_project_direction!,
        unsafe_project_variables!,
        unsafe_updatable_variables!,
        unsafe_update!
else
    using ..LoopVectorization
    using ..NumOptBase
    using ..NumOptBase:
        PlusOrMinus,
        SimdLoopEngine,
        TurboArray,
        TurboLoopEngine,
        get_bound,
        can_vary,
        only_arrays,
        project_direction,
        project_variable
    import ..NumOptBase:
        norm1,
        norm2,
        norminf,
        unsafe_inner,
        unsafe_map!,
        unsafe_project_direction!,
        unsafe_project_variables!,
        unsafe_updatable_variables!,
        unsafe_update!
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

    # Type of arguments suitable to represent a bound for `@turbo` optimized
    # loops.
    const TurboBound{T,N} = Union{Nothing,T,TurboArray{T,N}}

    @inline function unsafe_inner(::Type{<:TurboLoopEngine},
                                  x::TurboArray{T,N},
                                  y::TurboArray{T,N}) where {T<:HWReal,N}
        acc = inner(zero(eltype(x)), zero(eltype(y)))
        @turbo for i in eachindex(x, y)
            acc += inner(x[i], y[i])
        end
        return acc
    end

    @inline function unsafe_inner(::Type{<:TurboLoopEngine},
                                  w::TurboArray{T,N},
                                  x::TurboArray{T,N},
                                  y::TurboArray{T,N}) where {T<:HWReal,N}
        acc = inner(zero(eltype(w)), zero(eltype(x)), zero(eltype(y)))
        @turbo for i in eachindex(w, x, y)
            acc += inner(w[i], x[i], y[i])
        end
        return acc
    end

    @inline function norm1(::Type{<:TurboLoopEngine},
                           x::TurboArray{T,N}) where {T<:HWReal,N}
        acc = zero(eltype(x))
        @turbo for i in eachindex(x)
            acc += abs(x[i])
        end
        return acc
    end

    @inline function norm2(::Type{<:TurboLoopEngine},
                           x::TurboArray{T,N}) where {T<:HWReal,N}
        acc = zero(eltype(x))
        @turbo for i in eachindex(x)
            acc += abs2(x[i])
        end
        return sqrt(acc)
    end

    @inline function norminf(::Type{<:TurboLoopEngine},
                             x::TurboArray{T,N}) where {T<:HWReal,N}
        r = zero(eltype(x))
        @turbo for i in eachindex(x)
            a = abs(x[i])
            r = a > r ? a : r
        end
        return r
    end

    @inline function unsafe_map!(::Type{<:TurboLoopEngine},
                                 f::Function,
                                 dst::TurboArray{T,N},
                                 x::TurboArray{T,N}) where {T<:HWReal,N}
        if can_avx(f)
            @turbo for i in eachindex(dst, x)
                dst[i] = f(x[i])
            end
        else
            # Fallback to SIMD loop vectorization.
            unsafe_map!(SimdLoopEngine, f, dst, x)
        end
        return nothing
    end

    @inline function unsafe_map!(::Type{<:TurboLoopEngine},
                                 f::Function,
                                 dst::TurboArray{T,N},
                                 x::TurboArray{T,N},
                                 y::TurboArray{T,N}) where {T<:HWReal,N}
        if can_avx(f)
            @turbo for i in eachindex(dst, x, y)
                dst[i] = f(x[i], y[i])
            end
        else
            # Fallback to SIMD loop vectorization.
            unsafe_map!(SimdLoopEngine, f, dst, x, y)
        end
        return nothing
    end

    @inline function unsafe_map!(::Type{<:TurboLoopEngine},
                                 f::Function,
                                 dst::TurboArray{T,N},
                                 x::TurboArray{T,N},
                                 y::TurboArray{T,N},
                                 z::TurboArray{T,N}) where {T<:HWReal,N}
        if can_avx(f)
            @turbo for i in eachindex(dst, x, y, z)
                dst[i] = f(x[i], y[i], z[i])
            end
        else
            # Fallback to SIMD loop vectorization.
            unsafe_map!(SimdLoopEngine, f, dst, x, y, z)
        end
        return nothing
    end

    function unsafe_update!(::Type{<:TurboLoopEngine},
                            dst::TurboArray{T,N},
                            α::T,
                            x::TurboArray{T,N}) where {T<:HWReal,N}
        @turbo for i in eachindex(dst, x)
            dst[i] += α*x[i]
        end
        return nothing
    end

    function unsafe_update!(::Type{<:TurboLoopEngine},
                            dst::TurboArray{T,N},
                            α::T,
                            x::TurboArray{T,N},
                            y::TurboArray{T,N}) where {T<:HWReal,N}
        @turbo for i in eachindex(dst, x, y)
            dst[i] += α*x[i]*y[i]
        end
        return nothing
    end

    if false # FIXME: @turbo does not work yet

        function unsafe_project_variables!(::Type{<:TurboLoopEngine},
                                           dst::TurboArray{T,N},
                                           x::TurboArray{T,N},
                                           lower::TurboBound{T,N},
                                           upper::TurboBound{T,N}) where {T,N}
            @turbo for i in eachindex(dst, x, only_arrays(lower, upper)...)
                dst[i] = project_variable(x[i], get_bound(lower, i), get_bound(upper, i))
            end
            return nothing
        end

        function unsafe_project_direction!(::Type{<:TurboLoopEngine},
                                           dst::TurboArray{T,N},
                                           x::TurboArray{T,N},
                                           pm::PlusOrMinus,
                                           d::TurboArray{T,N},
                                           lower::TurboBound{T,N},
                                           upper::TurboBound{T,N}) where {T,N}
            @turbo for i in eachindex(dst, x, d, only_arrays(lower, upper)...)
                dst[i] = project_direction(x[i], pm, d[i], get_bound(lower, i), get_bound(upper, i))
            end
            return nothing
        end

        function unsafe_updatable_variables!(::Type{<:TurboLoopEngine},
                                             dst::TurboArray{B,N},
                                             x::TurboArray{T,N},
                                             pm::PlusOrMinus,
                                             d::TurboArray{T,N},
                                             lower::TurboBound{T,N},
                                             upper::TurboBound{T,N}) where {B,T,N}
            @turbo for i in eachindex(dst, x, d, only_arrays(lower, upper)...)
                dst[i] = can_vary(B, x[i], pm, d[i], get_bound(lower, i), get_bound(upper, i))
            end
            return nothing
        end

    end

else

    @warn """We are currently skipping improved optimizations by `LoopVectorization`
             because the `@turbo` macro is absent. This maybe due to your Julia
             version ($VERSION) being too old or to an outdated version of
             the `LoopVectorization` package. Please consider upgrading."""

end

end # module
