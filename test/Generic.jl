"""

Reference implementations (unoptimized) of methods and types for testing
package.

"""
module Generic

using ..NumOptBase: BoundedSet, PlusOrMinus, only_arrays

# Custom array type to check other versions than the ones that apply for
# strided arrays.
struct OtherArray{T,N,L,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    OtherArray(arr::A) where {T,N,A<:AbstractArray{T,N}} =
        new{T,N,IndexStyle(A)===IndexLinear(),A}(arr)
end
Base.parent(A::OtherArray) = A.parent
Base.length(A::OtherArray) = length(A.parent)
Base.size(A::OtherArray) = size(A.parent)
Base.axes(A::OtherArray) = axes(A.parent)
Base.IndexStyle(::Type{<:OtherArray{T,N,false}}) where {T,N} = IndexCartesian()
@inline function Base.getindex(A::OtherArray{T,N,false}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    return @inbounds getindex(parent(A), I...)
end
@inline function Base.setindex!(A::OtherArray{T,N,false}, x, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(parent(A), x, I...)
    return A
end
@inline Base.IndexStyle(::Type{<:OtherArray{T,N,true}}) where {T,N} = IndexLinear()
@inline function Base.getindex(A::OtherArray{T,N,true}, i::Int) where {T,N}
    @boundscheck checkbounds(A, i)
    return @inbounds getindex(parent(A), i)
end
@inline function Base.setindex!(A::OtherArray{T,N,true}, x, i::Int) where {T,N}
    @boundscheck checkbounds(A, i)
    @inbounds setindex!(parent(A), x, i)
    return A
end
Base.similar(A::OtherArray, ::Type{T}, dims::Dims) where {T} =
    OtherArray(similar(parent(A), T, dims))
Base.convert(::Type{T}, A::T) where {T<:OtherArray} = A
Base.convert(::Type{T}, A::AbstractArray) where {T<:OtherArray} = T(A)

# Reference methods (NOTE: muti-dimensional arrays are treated as vectors and
# complexes as pairs of reals).
norm1(x::Real) = abs(x)
norm1(x::Complex) = norm1(real(x)) + norm1(imag(x))
norm1(x::AbstractArray) =
    mapreduce(norm1, +, x; init = norm1(zero(eltype(x))))

norm2(x::Real) = abs(x)
norm2(x::Complex) = sqrt(abs2(x))
norm2(x::AbstractArray) =
    sqrt(mapreduce(abs2, +, x; init = abs2(zero(eltype(x)))))

norminf(x::Real) = abs(x)
norminf(x::Complex) = max(abs(real(x)), abs(imag(x)))
norminf(x::AbstractArray) =
    mapreduce(norminf, max, x; init = norminf(zero(eltype(x))))

inner(x::Real, y::Real) = x*y
inner(x::Complex, y::Complex) = real(x)*real(y) + imag(x)*imag(y)
inner(w::Real, x::Real, y::Real) = w*x*y
inner(x::AbstractArray, y::AbstractArray) =
    mapreduce(inner, +, x, y; init = inner(zero(eltype(x)), zero(eltype(y))))
inner(w::AbstractArray, x::AbstractArray, y::AbstractArray) =
    mapreduce(inner, +, w, x, y; init = inner(zero(eltype(w)), zero(eltype(x)), zero(eltype(y))))

"""
    Generic.same_float(x, y) -> bool

yields whether `x` and `y` have the same floating-point values. NaNs of any
kind are considered as the same floating-point value. Plus zero and minus zero
are considered as different floating-point values. Because of the former
assumption, `x === y` is not sufficient to test whether `x` and `y` have the
same floating-point values.

"""
same_float(x, y) = false
same_float(x::T, y::T) where {T <: AbstractFloat} =
    # Comparing results is tricky because not all NaNs are the same.
    (isnan(x) & isnan(y)) | (x === y)
function same_float(x::AbstractArray{T,N},
                    y::AbstractArray{T,N}; verb::Bool = false) where {T,N}
    @assert axes(x) == axes(y)
    result = true
    for i in CartesianIndices(x)
        flag = same_float(x[i], y[i])
        result &= flag
        if verb && !flag
            printstyled(stderr, "ERROR: $(x[i]) ≠ $(y[i]) at indices $(Tuple(i))\n";
                        color = :red)
        end
    end
    return result
end

lower_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L<:Nothing,U} = typemin(T)
lower_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L<:Number,U} = Ω.lower
lower_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L<:AbstractArray,U} = Ω.lower[i]

upper_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L,U<:Nothing} = typemax(T)
upper_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L,U<:Number} = Ω.upper
upper_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L,U<:AbstractArray} = Ω.upper[i]

# Reference version of project_variables!. Not meant to be smart, just to
# provide correct result.
function project_variables!(dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {T,N}
   for i in eachindex(dst, x, only_arrays(Ω.lower, Ω.upper)...)
        dst[i] = min(max(x[i], lower_bound(Ω, i)), upper_bound(Ω, i))
    end
    return dst
end

# Reference version of project_direction!. Not meant to be smart, just to
# provide correct result.
function project_direction!(dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {T,N}
    for i in eachindex(x, d, only_arrays(Ω.lower, Ω.upper)...)
        s = pm(d[i])
        unblocked =
            s < zero(s) ? lower_bound(Ω, i) < x[i] :
            s > zero(s) ? upper_bound(Ω, i) > x[i] : true
        dst[i] = unblocked ? d[i] : zero(T)
    end
    return dst
end

# Reference version of changing_variables!. Not meant to be smart, just to
# provide correct result.
function changing_variables!(dst::AbstractArray{B,N},
                             x::AbstractArray{T,N},
                             pm::PlusOrMinus, d::AbstractArray{T,N},
                             Ω::BoundedSet{T,N}) where {B,T,N}
    for i in eachindex(x, d, only_arrays(Ω.lower, Ω.upper)...)
        s = pm(d[i])
        unblocked =
            s < zero(s) ? lower_bound(Ω, i) < x[i] :
            s > zero(s) ? upper_bound(Ω, i) > x[i] : true
        dst[i] = unblocked ? one(B) : zero(B)
    end
    return dst
end

# Reference version of linesearch_limits. Not meant to be smart, just to
# provide correct result.
function linesearch_limits(x::AbstractArray{T,N},
                           pm::PlusOrMinus, d::AbstractArray{T,N},
                           Ω::BoundedSet{T,N}) where {T,N}
    amin = linesearch_stepmin(x, pm, d, Ω)
    amax = linesearch_stepmax(x, pm, d, Ω)
    return amin, amax
end

# Reference version of linesearch_stepmin. Not meant to be smart, just to
# provide correct result.
function linesearch_stepmin(x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {T,N}
    amin = typemax(T)
    for i in eachindex(x, d, only_arrays(Ω.lower, Ω.upper)...)
        s = pm(d[i])
        if s < zero(s)
            l = lower_bound(Ω, i)
            if l > typemin(l)
                amin = min(amin, (l - x[i])/s)
            end
        elseif s > zero(s)
            u = upper_bound(Ω, i)
            if u < typemax(u)
                amin = min(amin, (u - x[i])/s)
            end
        end
    end
    return amin::T
end

# Reference version of linesearch_stepmax. Not meant to be smart, just to
# provide correct result.
function linesearch_stepmax(x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {T,N}
    amax = T(NaN)
    for i in eachindex(x, d, only_arrays(Ω.lower, Ω.upper)...)
        s = pm(d[i])
        if s < zero(s)
            l = lower_bound(Ω, i)
            if l > typemin(l)
                a = (l - x[i])/s
                if isnan(amax) || zero(a) ≤ a < amax
                    amax = a
                end
            end
        elseif s > zero(s)
            u = upper_bound(Ω, i)
            if u < typemax(u)
                a = (u - x[i])/s
                if isnan(amax) || zero(a) ≤ a < amax
                    amax = a
                end
            end
        end
    end
    return (isnan(amax) ? T(Inf) : amax)::T
end

end # module Generic
