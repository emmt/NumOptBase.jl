# BoundSet constructors.
function BoundedSet(lower::AbstractArray{<:Any,N}, upper::AbstractArray{<:Any,N}) where {N}
    T = float(promote_type(eltype(lower), eltype(upper)))
    return BoundedSet{T}(lower, upper)
end

function BoundedSet{T}(lower::AbstractArray{<:Any,N}, upper::AbstractArray{<:Any,N}) where {T,N}
    return BoundedSet{T}(convert_eltype(T, lower), convert_eltype(T, upper))
end

function BoundedSet(vars::AbstractArray; kwds...)
    return BoundedSet{float(eltype(vars))}(vars; kwds...)
end

function BoundedSet{T}(vars::AbstractArray{<:Any,N};
                       lower = nothing, upper = nothing) where {T,N}
    rngs = axes(vars)
    lower = if lower isa AbstractArray
        check_axes("lower bound", lower, rngs)
        convert_eltype(T, lower)
    else
        val = lower isa Nothing ? typemin(T) : as(T, lower)
        UniformArray(val, rngs)
    end
    upper = if upper isa AbstractArray
        check_axes("upper bound", upper, rngs)
        convert_eltype(T, upper)
    else
        val = upper isa Nothing ? typemax(T) : as(T, upper)
        UniformArray(val, rngs)
    end
    return BoundedSet{T}(lower, upper)
end

# Conversion constructors.
BoundedSet(Ω::BoundedSet) = Ω
BoundedSet{T}(Ω::BoundedSet{T}) where {T} = Ω
BoundedSet{T}(Ω::BoundedSet) where {T} = BoundedSet(convert_eltype(T, Ω.lower), convert_eltype(T, Ω.upper))
BoundedSet{T,N}(Ω::BoundedSet{<:Any,N}) where {T,N} = Ω

Base.convert(::Type{BoundedSet}, Ω::BoundedSet) = BoundedSet{T}(Ω)
Base.convert(::Type{BoundedSet{T}}, Ω::BoundedSet) where {T} = BoundedSet{T}(Ω)
Base.convert(::Type{BoundedSet{T,N}}, Ω::BoundedSet) where {T,N} = BoundedSet{T,N}(Ω)
Base.convert(::Type{BoundedSet{T,N,L}}, Ω::BoundedSet) where {T,N,L<:AbstractArray{T,N}} =
    BoundedSet{T,N}(convert(L, Ω.lower), Ω.upper)
Base.convert(::Type{BoundedSet{T,N,L,U}}, Ω::BoundedSet) where {T,N,L<:AbstractArray{T,N},U<:AbstractArray{T,N}} =
    BoundedSet{T,N}(convert(L, Ω.lower), convert(U, Ω.upper))

Base.eltype(Ω::BoundedSet) = eltype(typeof(Ω))
Base.eltype(::Type{<:BoundedSet{T,N}}) where {T,N} = T

Base.ndims(Ω::BoundedSet) = ndims(typeof(Ω))
Base.ndims(::Type{<:BoundedSet{T,N}}) where {T,N} = N

Base.iterate(Ω::BoundedSet, state::Int=1) =
    state == 1 ? (Ω.lower, 2) :
    state == 2 ? (Ω.upper, 3) : nothing

Base.first(Ω::BoundedSet) = Ω.lower
Base.last(Ω::BoundedSet) = Ω.upper

# BoundSet API.
function Base.isempty(Ω::BoundedSet)
    below, above = is_bounding(Ω)
    if below & above
        return !feasible_bounds(Ω.lower, Ω.upper)
    elseif below
        return !feasible_bounds(Ω.lower, nothing)
    elseif above
        return !feasible_bounds(nothing, Ω.upper)
    else
        return !feasible_bounds(nothing, nothing)
    end
end

# Test that the bounds give a feasible set. According to IEEE rules for
# comparisons, it will be considered as empty if some bounds are NaN's.
feasible_bounds(lower::Nothing, upper::Nothing) = true
feasible_bounds(lower::AbstractArray, upper::Nothing) = any(isnan, lower)
feasible_bounds(lower::Nothing, upper::AbstractArray) = any(isnan, upper)
feasible_bounds(lower::AbstractArray, upper::AbstractArray) =
    axes(lower) == axes(upper) && unsafe_feasible_bounds(lower, upper)

unsafe_feasible_bounds(lower::AbstractArray, upper::AbstractArray) =
    mapreduce(≤, &, lower, upper; init=true)

unsafe_feasible_bounds(lower::AbstractArray, upper::AbstractUniformArray) =
    mapreduce(≤, &, lower, upper; init=true)

unsafe_feasible_bounds(lower::AbstractUniformArray, upper::AbstractArray) =
    # Swap operands so that the non-uniform array appears first. This is needed
    # for `mapreduce` to work with GPU arrays.
    mapreduce(≥, &, upper, lower; init=true)

unsafe_feasible_bounds(lower::AbstractUniformArray, upper::AbstractUniformArray) =
    value(lower) ≤ value(upper)

Base.in(x, Ω::BoundedSet) = false
function Base.in(x::AbstractArray{T,N}, Ω::BoundedSet{T,N}) where {T,N}
    l, u = Ω
    rngs = axes(x)
    axes(l) == rngs || return false
    axes(u) == rngs || return false
    below, above = is_bounding(l, u)
    if below & above
        # Test that the bounded set is feasible. According to IEEE rules for
        # comparisons, it will be considered as empty if some bounds are NaN's.
        return mapreduce(in_bounds, &, x, l, u; init=true)
    elseif below
        return mapreduce(≥, &, x, l; init=true)
    elseif above
        return mapreduce(≤, &, x, u; init=true)
    else
        return true
    end
end

in_bounds(x::T, l::T, u::T) where {T} = ((x ≥ l)&(x ≤ u))

# Projector API.
(P::Projector)(x::AbstractArray) = P(similar(x), x)
(P::Projector)(dst::AbstractArray, src::AbstractArray) =
    project_variables!(dst, src, P.Ω)

"""
    NumOptBase.is_bounding_below(l) -> bool

yields whether lower bound set by `l` may be limiting. For efficiency, if `l`
is multi-valued, it is considered as limiting even though all its values may
all be equal to `-∞`.

"""
is_bounding_below(l::Nothing) = false
is_bounding_below(l::AbstractArray) = true
is_bounding_below(l::Number) = l > typemin(l)
is_bounding_below(l::AbstractUniformArray) = is_bounding_below(value(l))
is_bounding_below(Ω::BoundedSet) = is_bounding_below(Ω.lower)

"""
    NumOptBase.is_bounding_above(u) -> bool

yields whether upper bound set by `u` may be limiting. For efficiency, if `u`
is multi-valued, it is considered as limiting even though all its values may
all be equal to `+∞`.

"""
is_bounding_above(u::Nothing) = false
is_bounding_above(u::AbstractArray) = true
is_bounding_above(u::Number) = u < typemax(u)
is_bounding_above(u::AbstractUniformArray) = is_bounding_above(value(u))
is_bounding_above(Ω::BoundedSet) = is_bounding_above(Ω.upper)

"""
   NumOptBase.is_bounding(lower, upper) -> below, above
   NumOptBase.is_bounding(Ω::NumOptBase.BoundedSet) -> below, above

yield whether bounds set by `lower` and `upper` or by `Ω` may be limiting below
and/or above. See [`NumOptBase.is_bounding_below`](@ref),
[`NumOptBase.is_bounding_above`](@ref), and [`NumOptBase.BoundedSet`](@ref).

"""
is_bounding(lower, upper) = (is_bounding_below(lower), is_bounding_above(upper))
is_bounding(Ω::BoundedSet) = is_bounding(Ω...)

"""
    project_variables!([E,] dst, x, Ω) -> dst

overwrites destination `dst` with the *projected variables* such that:

    dst = P(x) = argmin_{y ∈ Ω} ‖y - x‖²

where `P` is the projection onto the feasible set `Ω` and `x` are the
variables.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, Ω...)` is assumed.

"""
function project_variables!(dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {T,N}
    return project_variables!(engine(dst, x, Ω...), dst, x, Ω)
end

function project_variables!(::Type{E},
                            dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {E<:Engine,T,N}
    # Check arguments and directly handle the unbounded case. If there are any
    # bounds, call an unsafe method with non-limiting bounds specified as
    # `nothing` so as to allow for dispatching on an optimized version based on
    # the bound types.
    check_axes(x; dest=dst, lower=Ω.lower, upper=Ω.upper)
    below, above = is_bounding(Ω)
    if below & above
        unsafe_project_variables!(E, dst, x, Ω.lower, Ω.upper)
    elseif below
        unsafe_project_variables!(E, dst, x, Ω.lower, nothing)
    elseif above
        unsafe_project_variables!(E, dst, x, nothing, Ω.upper)
    elseif dst !== x
        unsafe_copy!(dst, x)
    end
    return dst
end

"""
    project_direction!([E,] dst, x, ±, d, Ω) -> dst

overwrites destination `dst` with the *projected direction* such that, provided
`x ∈ Ω`, then:

    ∀ α ∈ [0,ε], P(x ± α⋅d) = x ± α⋅dst

for some `ε > 0` and where `P` is the projection onto the feasible set
`Ω ⊆ ℝⁿ`, `x ∈ Ω` are the variables, and `d ∈ ℝⁿ` is a search direction.

For efficiency, it is assumed without checking that `x ∈ Ω` holds.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, d, Ω...)` is assumed.

"""
function project_direction!(dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            pm::PlusOrMinus,
                            d::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {T,N}
    return project_direction!(engine(dst, x, d, Ω...), dst, x, pm, d, Ω)
end

function project_direction!(::Type{E},
                            dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            pm::PlusOrMinus,
                            d::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {E<:Engine,T,N}
    # NOTE: See comments in `project_variables!`.
    check_axes(x; dest=dst, dir=d, lower=Ω.lower, upper=Ω.upper)
    below, above = is_bounding(Ω)
    if below & above
        unsafe_project_direction!(E, dst, x, pm, d, Ω.lower, Ω.upper)
    elseif below
        unsafe_project_direction!(E, dst, x, pm, d, Ω.lower, nothing)
    elseif above
        unsafe_project_direction!(E, dst, x, pm, d, nothing, Ω.upper)
    elseif dst !== d
        unsafe_copy!(dst, d)
    end
    return dst
end

"""
    updatable_variables!([E,] dst, x, ±, d, Ω) -> dst

overwrites destination `dst` with ones where variables in `x` can be updated
within the bounds implemented by `Ω` along direction `±d` and zeros elsewhere.

For efficiency, it is assumed without checking that `x ∈ Ω` holds.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, d, Ω...)` is assumed.

It is assumed that `x[i]` is not updatable if `d[i] = 0`. Hence, if `±d =
-∇f(x)`, with `∇f(x)` the gradient of the objective function, the destination
is set to zero everywhere the Karush-Kuhn-Tucker (K.K.T.) conditions are
satisfied for the problem:

    minₓ f(x) s.t. x ∈ Ω

In other words, `all(izero, dst)` holds for (exact) convergence.

"""
function updatable_variables!(dst::AbstractArray{<:Real,N},
                              x::AbstractArray{T,N},
                              pm::PlusOrMinus,
                              d::AbstractArray{T,N},
                              Ω::BoundedSet{T,N}) where {T,N}
    return updatable_variables!(engine(dst, x, d, Ω...), dst, x, pm, d, Ω)
end

function updatable_variables!(::Type{E},
                              dst::AbstractArray{<:Real,N},
                              x::AbstractArray{T,N},
                              pm::PlusOrMinus,
                              d::AbstractArray{T,N},
                              Ω::BoundedSet{T,N}) where {E<:Engine,T,N}
    # NOTE: See comments in `project_variables!`.
    check_axes(x; dest=dst, dir=d, lower=Ω.lower, upper=Ω.upper)
    below, above = is_bounding(Ω)
    if below & above
        unsafe_updatable_variables!(E, dst, x, pm, d, Ω.lower, Ω.upper)
    elseif below
        unsafe_updatable_variables!(E, dst, x, pm, d, Ω.lower, nothing)
    elseif above
        unsafe_updatable_variables!(E, dst, x, pm, d, nothing, Ω.upper)
    else
        fill!(dst, one(eltype(dst)))
    end
    return dst
end

"""
    linesearch_limits([E,] x₀, ±, d, Ω) -> αₘᵢₙ, αₘₐₓ

yields the limits `αₘᵢₙ` and `αₘₐₓ` for the step length `α ≥ 0` in a
line-search where iterates `x` are given by:

      x = P(x₀ ± α⋅d)

where `P(x)` denotes the orthogonal projection on the convex set `Ω` and where `±`
is either `+` or `-`.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, d, Ω...)` is assumed.

Output value `αₘᵢₙ` is the greatest nonnegative step length such that:

    0 ≤ α ≤ αₘᵢₙ  ⟹  P(x₀ ± α⋅d) = x₀ ± α⋅d

Output value `αₘₐₓ` is the least nonnegative step length such that:

    α ≥ αₘₐₓ ≥ 0  ⟹  P(x₀ ± α⋅d) = P(x₀ ± αₘₐₓ⋅d)

In other words, no bounds are overcome if `0 ≤ α ≤ αₘᵢₙ` and the projected
variables are all the same for any `α` such that `α ≥ αₘₐₓ ≥ 0`.

!!! warn
    For efficiency, it is assumed without checking that `x₀` is feasible, that
    is `x₀ ∈ Ω` holds.

See also [`linesearch_stepmin`](@ref) or [`linesearch_stepmax`](@ref) if only
one of `αₘᵢₙ` or `αₘₐₓ` is needed.

"""
linesearch_limits

"""
    linesearch_stepmin([E,] x₀, ±, d, Ω) -> αₘᵢₙ

yields the greatest nonnegative step length `αₘᵢₙ` such that:

    0 ≤ α ≤ αₘᵢₙ  ⟹  P(x₀ ± α⋅d) = x₀ ± α⋅d

where `P(x)` denotes the orthogonal projection on the convex set `Ω` and where
`±` is either `+` or `-`.

See [`linesearch_limits`](@ref) for details.

"""
linesearch_stepmin

"""
    linesearch_stepmax([E,] x₀, ±, d, Ω) -> αₘₐₓ

yields the least nonnegative step length `αₘₐₓ` such that:

    α ≥ αₘₐₓ ≥ 0  ⟹  P(x₀ ± α⋅d) = P(x₀ ± αₘₐₓ⋅d)

where `P(x)` denotes the orthogonal projection on the convex set `Ω` and where
`±` is either `+` or `-`.

See [`linesearch_limits`](@ref) for details.

"""
linesearch_stepmax

for func in (:linesearch_limits, :linesearch_stepmin, :linesearch_stepmax)
    unsafe_func = Symbol("unsafe_",func)
    @eval begin
        function $func(x::AbstractArray{T,N},
                       pm::PlusOrMinus,
                       d::AbstractArray{T,N},
                       Ω::BoundedSet{T,N}) where {T,N}
            return $func(engine(x, d, Ω...), x, pm, d, Ω)
        end

        function $func(::Type{E},
                       x::AbstractArray{T,N},
                       pm::PlusOrMinus,
                       d::AbstractArray{T,N},
                       Ω::BoundedSet{T,N}) where {E<:Engine,T,N}
            # NOTE: See comments in `project_variables!`.
            check_axes(x; dir=d, lower=Ω.lower, upper=Ω.upper)
            below, above = is_bounding(Ω)
            if below & above
                $unsafe_func(E, x, pm, d, Ω.lower, Ω.upper)
            elseif below
                $unsafe_func(E, x, pm, d, Ω.lower, nothing)
            elseif above
                $unsafe_func(E, x, pm, d, nothing, Ω.upper)
            else
                unconstrained_result($func, T)
            end
        end
    end
end

unconstrained_result(::typeof(linesearch_stepmin), ::Type{T}) where {T} = typemax(T)
unconstrained_result(::typeof(linesearch_stepmax), ::Type{T}) where {T} = typemax(T)
unconstrained_result(::typeof(linesearch_limits), ::Type{T}) where {T} =
    (unconstrained_result(linesearch_stepmin, T),
     unconstrained_result(linesearch_stepmax, T))

# Unsafe implementations for basic engines.
for (optim, array, engine) in ((:none,      AbstractArray, LoopEngine),
                               (:inbounds,  AbstractArray, InBoundsLoopEngine),
                               (:simd,      SimdArray,     SimdLoopEngine))
    @eval begin
        function unsafe_project_variables!(::Type{<:$engine},
                                           dst::AbstractArray{T,N},
                                           x::AbstractArray{T,N},
                                           lower, upper) where {T,N}
            @vectorize $optim for i in eachindex(dst, x, only_arrays(lower, upper)...)
                dst[i] = project_variable(x[i], get_bound(lower, i), get_bound(upper, i))
            end
            return nothing
        end
        function unsafe_project_direction!(::Type{<:$engine},
                                           dst::AbstractArray{T,N},
                                           x::AbstractArray{T,N},
                                           pm::PlusOrMinus,
                                           d::AbstractArray{T,N},
                                           lower, upper) where {T,N}
            @vectorize $optim for i in eachindex(dst, x, d, only_arrays(lower, upper)...)
                dst[i] = project_direction(x[i], pm, d[i], get_bound(lower, i), get_bound(upper, i))
            end
            return nothing
        end
        function unsafe_updatable_variables!(::Type{<:$engine},
                                             dst::AbstractArray{B,N},
                                             x::AbstractArray{T,N},
                                             pm::PlusOrMinus,
                                             d::AbstractArray{T,N},
                                             lower, upper) where {B,T,N}
            @vectorize $optim for i in eachindex(dst, x, d, only_arrays(lower, upper)...)
                dst[i] = can_vary(B, x[i], pm, d[i], get_bound(lower, i), get_bound(upper, i))
            end
            return nothing
        end
        function unsafe_linesearch_limits(::Type{<:$engine},
                                          x::AbstractArray{T,N},
                                          pm::PlusOrMinus,
                                          d::AbstractArray{T,N},
                                          lower, upper) where {T,N}
            αmin, αmax = initial_stepmin(T), initial_stepmax(T)
            @vectorize $optim for i in eachindex(x, d, only_arrays(lower, upper)...)
                αmin, αmax = update_limits(αmin, αmax, x[i], pm, d[i],
                                           get_bound(lower, i), get_bound(upper, i))
            end
            return final_stepmin(αmin), final_stepmax(αmax)
        end
        function unsafe_linesearch_stepmin(::Type{<:$engine},
                                           x::AbstractArray{T,N},
                                           pm::PlusOrMinus,
                                           d::AbstractArray{T,N},
                                           lower, upper) where {T,N}
            αmin = initial_stepmin(T)
            @vectorize $optim for i in eachindex(x, d, only_arrays(lower, upper)...)
                αmin = update_stepmin(αmin, x[i], pm, d[i],
                                      get_bound(lower, i), get_bound(upper, i))
            end
            return final_stepmin(αmin)
        end
        function unsafe_linesearch_stepmax(::Type{<:$engine},
                                           x::AbstractArray{T,N},
                                           pm::PlusOrMinus,
                                           d::AbstractArray{T,N},
                                           lower, upper) where {T,N}
            αmax = initial_stepmax(T)
            @vectorize $optim for i in eachindex(x, d, only_arrays(lower, upper)...)
                αmax = update_stepmax(αmax, x[i], pm, d[i],
                                      get_bound(lower, i), get_bound(upper, i))
            end
            return final_stepmax(αmax)
        end
    end
end

"""
    NumOptBase.get_bound(B, i)

yields the bound value at index `i` for object `B` implementing lower or
upper bounds. This helper function is to simplify code, it yields `B[i]` if `B`
is an array and `B` otherwise. Bound checking is propagated.

"""
get_bound(B, i) = B
@inline function get_bound(B::AbstractArray{<:Any,N}, i::CartesianIndex{N}) where {N}
    I = Tuple(i)
	@boundscheck checkbounds(B, I...)
	return @inbounds B[I...]
end
@inline function get_bound(B::AbstractArray, i::Integer)
	@boundscheck checkbounds(B, i)
	return @inbounds B[i]
end

# Scalar methods to project a variable or a search direction and to check
# whether a variable is blocked.
#
# These methods are intended to be suitable for any type of arrays. We avoid
# branching (hence use `ifelse`, `|`, or `&` instead of `?:`, `||`, or `&&`) to
# not prevent loop vectorization and we very much rely on Julia optimizer to
# simplify expressions depending on the type of the bounds. All these methods
# are supposed to be in-lined.

project_variable(x, lower, upper) = project_above(project_below(x, lower), upper)

project_below(x, lower::Nothing) = x
project_above(x, upper::Nothing) = x

project_below(x::T, lower::T) where {T} = x < lower ? lower : x
project_above(x::T, upper::T) where {T} = x > upper ? upper : x

project_direction(x, pm::PlusOrMinus, d, lower, upper) =
    ifelse(can_vary(x, pm, d, lower, upper), d, zero(d))

"""
    NumOptBase.can_vary([T=Bool,] x, ±, d, lower, upper)

yields whether `x ± α⋅d != x` can hold within the `lower` and `upper` bounds
and for some `α > 0`. Optional argument `T` is the type of the result:
`oneunit(T)` if true, `zero(T)` otherwise. `lower` and/or `upper` may be
`nothing` if there is no such bound.

If no variables can vary in the direction `±d = -∇f(x)`, then the
Karush-Kuhn-Tucker (K.K.T.) conditions hold for the bound constrained
minimization of `f(x)`.

"""
can_vary(::Type{T}, x, pm::PlusOrMinus, d, lower, upper) where {T} =
    ifelse(can_vary(x, pm, d, lower, upper), oneunit(T), zero(T))
can_vary(x, pm::PlusOrMinus, d, lower, upper) =
    can_decrease(x, pm, d, lower) |
    can_increase(x, pm, d, upper)

# Yield whether `lower ≤ x ± α⋅d < x` is possible for `α > 0`.
can_decrease(x, pm::PlusOrMinus, d, lower) = is_negative(pm, d) & (x > lower)
can_decrease(x, pm::PlusOrMinus, d, lower::Nothing) = is_negative(pm, d)

# Yield whether `x < x ± α⋅d ≤ upper` is possible for `α > 0`.
can_increase(x, pm::PlusOrMinus, d, upper) = is_positive(pm, d) & (x < upper)
can_increase(x, pm::PlusOrMinus, d, upper::Nothing) = is_positive(pm, d)

is_positive(         x) = x > zero(x)
is_positive(::Plus,  x) = is_positive(x)
is_positive(::Minus, x) = is_negative(x)

is_negative(         x) = x < zero(x)
is_negative(::Plus,  x) = is_negative(x)
is_negative(::Minus, x) = is_positive(x)

# NOTE: The following code relies on IEEE arithmetics to compute the
#       line-search step limits. In particular, it is assumed that comparisons
#       can only be true if none of the operands is NaN. Do not change this
#       code unless you know what you do.
#
# step_to_bound(x,±,d,b) yields the value of the step α to reach the bound b,
# may yield NaN, infinite, or negative value.
step_to_bound(x::T, ::Plus,  d::T, b::T) where {T} = (b - x)/d
step_to_bound(x::T, ::Minus, d::T, b::T) where {T} = (x - b)/d
#
# Initialize, update, and finalize αmin.
initial_stepmin(::Type{T}) where {T} = typemax(T)
update_stepmin(αmin::T, α::T) where {T} = zero(α) ≤ α < αmin ? α : αmin
final_stepmin(αmin) = αmin
#
# Initialize, update, and finalize αmax.
initial_stepmax(::Type{T}) where {T} = -one(T)
update_stepmax(αmax::T, α::T) where {T} = α > αmax ? α : αmax
final_stepmax(αmax) = αmax ≥ zero(αmax) ? αmax : typemax(αmax)

# NOTE: The `update_...` methods are written so as to avoid branches, except
#       those based on types which should be optimized out by the compiler.

function update_limits(αmin::T, αmax::T, x::T, pm::PlusOrMinus, d::T,
                       lower::L, upper::U) where {T,
                                                  L<:Union{T,Nothing},
                                                  U<:Union{T,Nothing}}
    if L === T
        α = step_to_bound(x, pm, d, lower)
        αmin = update_stepmin(αmin, α)
        αmax = update_stepmax(αmax, α)
    end
    if U === T
        α = step_to_bound(x, pm, d, upper)
        αmin = update_stepmin(αmin, α)
        αmax = update_stepmax(αmax, α)
    end
    return αmin, αmax
end

function update_stepmin(αmin::T, x::T, pm::PlusOrMinus, d::T,
                        lower::L, upper::U) where {T,
                                                   L<:Union{T,Nothing},
                                                   U<:Union{T,Nothing}}
    if L === T
        α = step_to_bound(x, pm, d, lower)
        αmin = update_stepmin(αmin, α)
    end
    if U === T
        α = step_to_bound(x, pm, d, upper)
        αmin = update_stepmin(αmin, α)
    end
    return αmin
end

function update_stepmax(αmax::T, x::T, pm::PlusOrMinus, d::T,
                        lower::L, upper::U) where {T,
                                                   L<:Union{T,Nothing},
                                                   U<:Union{T,Nothing}}
    if L === T
        α = step_to_bound(x, pm, d, lower)
        αmax = update_stepmax(αmax, α)
    end
    if U === T
        α = step_to_bound(x, pm, d, upper)
        αmax = update_stepmax(αmax, α)
    end
    return αmax
end
