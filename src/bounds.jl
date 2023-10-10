# The search direction may be negated, this is indicated by argument `pm` which
# is either a `+` or a `-`.
const Plus = typeof(+)
const Minus = typeof(-)
const PlusOrMinus = Union{Plus,Minus}

"""
    NumOptBase.is_bounded_below(b) -> bool

yields whether lower bound set by `b` may be limiting. For efficiency, if `b`
is multi-valued, it is considered as limiting even though all its values may be
equal to `-∞`.

"""
is_bounded_below(b::Nothing) = false
is_bounded_below(b::Number) = b > typemin(b)
is_bounded_below(b::AbstractArray) = true
is_bounded_below(b::AbstractUniformArray) = is_bounded_below(StructuredArrays.getval(b))

"""
    NumOptBase.is_bounded_above(b) -> bool

yields whether upper bound set by `b` may be limiting. For efficiency, if `b`
is multi-valued, it is considered as limiting even though all its values may be
equal to `+∞`.

"""
is_bounded_above(b::Nothing) = false
is_bounded_above(b::Real) = b < typemax(b)
is_bounded_above(b::AbstractArray) = true
is_bounded_above(b::AbstractUniformArray) = is_bounded_above(StructuredArrays.getval(b))

"""
    NumOptBase.is_bounded(Ω::NumOptBase.BoundedSet) -> below, above

yields whether bounds set by `Ω` may be limiting below and/or above. See
[`NumOptBase.is_bounded_below`](@ref), [`NumOptBase.is_bounded_above`](@ref),
and [`NumOptBase.BoundedSet`](@ref).

"""
is_bounded(Ω::BoundedSet) = (is_bounded_below(Ω.lower),
                             is_bounded_above(Ω.upper))

"""
    project_variables!([E,] dst, x, Ω) -> dst

overwrites destination `dst` with the *projected variables* such that:

    dst = P(x) = argmin ‖y - x‖²   s.t.   y ∈ Ω

where `P` is the projection onto the feasible set `Ω` and `x` are the
variables.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, Ω)` is assumed.

"""
function project_variables!(dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {T<:Real,N}
    return project_variables!(
        engine(dst, x, only_arrays(Ω.lower, Ω.upper)...), dst, x, Ω)
end

function project_variables!(::Type{E},
                            dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {E<:Engine,T<:Real,N}
    # Check arguments and directly handle the unbounded case. If there are any
    # bounds, call an unsafe method with non-limiting bounds specified as
    # `nothing` so as to allow for dispatching on an optimized version based on
    # the bound types.
    check_axes(dst, x, only_arrays(Ω.lower, Ω.upper)...)
    below, above = is_bounded(Ω)
    if below & above
        unsafe_project_variables!(E, dst, x, Ω.lower, Ω.upper)
    elseif below
        unsafe_project_variables!(E, dst, x, Ω.lower, nothing)
    elseif above
        unsafe_project_variables!(E, dst, x, nothing, Ω.upper)
    elseif dst !== x
        copyto!(dst, x)
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

For efficiency, it is not checked that `x ∈ Ω` holds.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, d, Ω)` is assumed.

"""
function project_direction!(dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            pm::PlusOrMinus,
                            d::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {T<:Real,N}
    return project_direction!(
        engine(dst, x, d, only_arrays(Ω.lower, Ω.upper)...), dst, x, pm, d, Ω)
end

function project_direction!(::Type{E},
                            dst::AbstractArray{T,N},
                            x::AbstractArray{T,N},
                            pm::PlusOrMinus,
                            d::AbstractArray{T,N},
                            Ω::BoundedSet{T,N}) where {E<:Engine,T<:Real,N}
    # NOTE: See comments in `project_variables!`.
    check_axes(dst, x, d, only_arrays(Ω.lower, Ω.upper)...)
    below, above = is_bounded(Ω)
    if below & above
        unsafe_project_direction!(E, dst, x, pm, d, Ω.lower, Ω.upper)
    elseif below
        unsafe_project_direction!(E, dst, x, pm, d, Ω.lower, nothing)
    elseif above
        unsafe_project_direction!(E, dst, x, pm, d, nothing, Ω.upper)
    elseif dst !== d
        copyto!(dst, d)
    end
    return dst
end


"""
    unblocked_variables!([E,] dst, x, ±, d, Ω) -> dst

overwrites destination `dst` with ones where variables in `x` are not blocked
by the bounds implemented by `Ω` along direction `±d` and zeros elsewhere.

For efficiency, it is not checked that `x ∈ Ω` holds.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, d, Ω)` is assumed.

Note that, if `±d = -∇f(x)`, with `∇f(x)` the gradient of the objective
function, the destination is set to zero everywhere the Karush-Kuhn-Tucker
(K.K.T.) conditions are satisfied for the problem:

    minₓ f(x) s.t. x ∈ Ω

This can be used to check for (exact) convergence.

"""
function unblocked_variables!(dst::AbstractArray{<:Real,N},
                              x::AbstractArray{T,N},
                              pm::PlusOrMinus,
                              d::AbstractArray{T,N},
                              Ω::BoundedSet{T,N}) where {T<:Real,N}
    return unblocked_variables!(
        engine(dst, x, d, only_arrays(Ω.lower, Ω.upper)...), dst, x, pm, d, Ω)
end

function unblocked_variables!(::Type{E},
                              dst::AbstractArray{<:Real,N},
                              x::AbstractArray{T,N},
                              pm::PlusOrMinus,
                              d::AbstractArray{T,N},
                              Ω::BoundedSet{T,N}) where {E<:Engine,T<:Real,N}
    # NOTE: See comments in `project_variables!`.
    check_axes(dst, x, d, only_arrays(Ω.lower, Ω.upper)...)
    below, above = is_bounded(Ω)
    if below & above
        unsafe_unblocked_variables!(E, dst, x, pm, d, Ω.lower, Ω.upper)
    elseif below
        unsafe_unblocked_variables!(E, dst, x, pm, d, Ω.lower, nothing)
    elseif above
        unsafe_unblocked_variables!(E, dst, x, pm, d, nothing, Ω.upper)
    else
        fill!(dst, one(eltype(dst)))
    end
    return dst
end

"""
    linesearch_limits([E,] x₀, ±, d, Ω) -> αₘᵢₙ, αₘₐₓ

yields the limits `αₘᵢₙ` and `αₘₐₓ` for the step length `α` in a line-search
where iterates `x` are given by:

      x = P(x₀ ± α⋅d)

where `P(x)` denotes the orthogonal projection on the convex set `Ω` and where `±`
is either `+` or `-`.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, d, Ω)` is assumed.

Output value `αₘᵢₙ` is the greatest nonnegative step length such that:

    α ≤ αₘᵢₙ  ⟹  P(x₀ ± α⋅d) = x₀ ± α⋅d

Output value `αₘₐₓ` is the least nonnegative step length such that:

    α ≥ αₘₐₓ  ⟹  P(x₀ ± α⋅d) = P(x₀ ± αₘₐₓ⋅d)

In other words, no bounds are overcome if `0 ≤ α ≤ αₘᵢₙ` and the projected
variables are all the same for any `α` such that `α ≥ αₘₐₓ ≥ 0`.

!!! warn
    For efficiency, it is assumed without checking that `x₀` is feasible, that
    is `x₀ ∈ Ω` holds.

See also [`linesearch_min_step`](@ref) or [`linesearch_max_step`](@ref) if only
one of `αₘᵢₙ` or `αₘₐₓ` is needed.

"""
linesearch_limits

"""
    linesearch_min_step([E,] x₀, ±, d, Ω) -> αₘᵢₙ

yields the greatest nonnegative step length `αₘᵢₙ` such that:

    α ≤ αₘᵢₙ  ⟹  P(x₀ ± α⋅d) = x₀ ± α⋅d

where `P(x)` denotes the orthogonal projection on the convex set `Ω` and where `±`
is either `+` or `-`.

See [`linesearch_limits`](@ref) for details.

"""
linesearch_min_step

"""
    linesearch_max_step([E,] x₀, ±, d, Ω) -> αₘₐₓ

yields the least nonnegative step length `αₘₐₓ` such that:

    α ≥ αₘₐₓ  ⟹  P(x₀ ± α⋅d) = P(x₀ ± αₘₐₓ⋅d)

where `P(x)` denotes the orthogonal projection on the convex set `Ω` and where `±`
is either `+` or `-`.

See [`linesearch_limits`](@ref) for details.

"""
linesearch_max_step

for func in (:linesearch_limits, :linesearch_min_step, :linesearch_max_step)
    unsafe_func = Symbol("unsafe_",func)
    @eval begin
        function $func(x::AbstractArray{T,N},
                       pm::PlusOrMinus,
                       d::AbstractArray{T,N},
                       Ω::BoundedSet{T,N}) where {T<:Real,N}
            return $func(
                engine(x, d, only_arrays(Ω.lower, Ω.upper)...), x, pm, d, Ω)
        end

        function $func(::Type{E},
                       x::AbstractArray{T,N},
                       pm::PlusOrMinus,
                       d::AbstractArray{T,N},
                       Ω::BoundedSet{T,N}) where {E<:Engine,T<:Real,N}
            # NOTE: See comments in `project_variables!`.
            check_axes(x, d, only_arrays(Ω.lower, Ω.upper)...)
            below, above = is_bounded(Ω)
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

unconstrained_result(::typeof(linesearch_min_step), ::Type{T}) where {T} = typemax(T)
unconstrained_result(::typeof(linesearch_max_step), ::Type{T}) where {T} = typemax(T)
unconstrained_result(::typeof(linesearch_limits), ::Type{T}) where {T} =
    (unconstrained_result(linesearch_min_step, T),
     unconstrained_result(linesearch_max_step, T))

# Unsafe implementations for basic engines.
for (optim, array, engine) in ((:none,      AbstractArray, LoopEngine),
                               (:inbounds,  AbstractArray, InBoundsLoopEngine),
                               (:simd,      SimdArray,     SimdLoopEngine))
    @eval begin
        function unsafe_project_variables!(::Type{<:$engine},
                                           dst::AbstractArray{T,N},
                                           x::AbstractArray{T,N},
                                           lower::Bound{T,N},
                                           upper::Bound{T,N}) where {T,N}
            @vectorize $optim for i in eachindex(dst, x, only_arrays(lower, upper)...)
                dst[i] = project(x[i], get_bound(lower, i), get_bound(upper, i))
            end
            return nothing
        end
        function unsafe_project_direction!(::Type{<:$engine},
                                           dst::AbstractArray{T,N},
                                           x::AbstractArray{T,N},
                                           pm::PlusOrMinus,
                                           d::AbstractArray{T,N},
                                           lower::Bound{T,N},
                                           upper::Bound{T,N}) where {T,N}
            @vectorize $optim for i in eachindex(dst, x, d, only_arrays(lower, upper)...)
                dst[i] = project(x[i], pm, d[i], get_bound(lower, i), get_bound(upper, i))
            end
            return nothing
        end
        function unsafe_unblocked_variables!(::Type{<:$engine},
                                             dst::AbstractArray{B,N},
                                             x::AbstractArray{T,N},
                                             pm::PlusOrMinus,
                                             d::AbstractArray{T,N},
                                             lower::Bound{T,N},
                                             upper::Bound{T,N}) where {B,T,N}
            @vectorize $optim for i in eachindex(dst, x, d, only_arrays(lower, upper)...)
                dst[i] = is_unblocked(B, x[i], pm, d[i],
                                      get_bound(lower, i), get_bound(upper, i))
            end
            return nothing
        end
        function unsafe_linesearch_limits(::Type{<:$engine},
                                          x::AbstractArray{T,N},
                                          pm::PlusOrMinus,
                                          d::AbstractArray{T,N},
                                          lower::Bound{T,N},
                                          upper::Bound{T,N}) where {T,N}
            αmin, αmax = initial_min_step(T), initial_max_step(T)
            @vectorize $optim for i in eachindex(x, d, only_arrays(lower, upper)...)
                αmin, αmax = update_limits(αmin, αmax, x[i], pm, d[i],
                                           get_bound(lower, i), get_bound(upper, i))
            end
            return final_min_step(αmin), final_max_step(αmax)
        end
        function unsafe_linesearch_min_step(::Type{<:$engine},
                                            x::AbstractArray{T,N},
                                            pm::PlusOrMinus,
                                            d::AbstractArray{T,N},
                                            lower::Bound{T,N},
                                            upper::Bound{T,N}) where {T,N}
            αmin = initial_min_step(T)
            @vectorize $optim for i in eachindex(x, d, only_arrays(lower, upper)...)
                αmin = update_min_step(αmin, x[i], pm, d[i],
                                       get_bound(lower, i), get_bound(upper, i))
            end
            return final_min_step(αmin)
        end
        function unsafe_linesearch_max_step(::Type{<:$engine},
                                            x::AbstractArray{T,N},
                                            pm::PlusOrMinus,
                                            d::AbstractArray{T,N},
                                            lower::Bound{T,N},
                                            upper::Bound{T,N}) where {T,N}
            αmax = initial_max_step(T)
            @vectorize $optim for i in eachindex(x, d, only_arrays(lower, upper)...)
                αmax = update_max_step(αmax, x[i], pm, d[i],
                                       get_bound(lower, i), get_bound(upper, i))
            end
            return final_max_step(αmax)
        end
    end
end

"""
    NumOptBase.get_bound(B, i)

yields a the bound value at index `i` for object `B` implementing lower or
upper bounds. This helper function is to simplify code, it yields `B[i]` if `B`
is an array and `B` otherwise. Bound checking is propagated.

"""
@inline get_bound(B::Union{Real,Nothing}, i) = B
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

project(x::Real, lower, upper) =
    project_above(project_below(x, lower), upper)

project(x::Real, pm::PlusOrMinus, d::Real, lower, upper) =
    ifelse(is_unblocked(x, pm, d, lower, upper), d, zero(d))

is_unblocked(x::Real, pm::PlusOrMinus, d::Real, lower, upper) =
    (is_unblocked_below(x, pm, d, lower) &
     is_unblocked_above(x, pm, d, upper))

is_unblocked(::Type{T}, x::Real, pm::PlusOrMinus, d::Real, lower, upper) where {T} =
    ifelse(is_unblocked(x, pm, d, lower, upper), one(T), zero(T))

project_below(x::T, lower::T) where {T<:Real} = x < lower ? lower : x
project_below(x::T, lower::Nothing) where {T<:Real} = x

project_above(x::T, upper::T) where {T<:Real} = x > upper ? upper : x
project_above(x::T, upper::Nothing) where {T<:Real} = x

is_unblocked_below(x::Real, pm::PlusOrMinus, d::Real, lower::Nothing) = true
is_unblocked_below(x::Real, pm::PlusOrMinus, d::Real, lower::Real) =
    (x > lower) | is_positive(pm, d)

is_unblocked_above(x::Real, pm::PlusOrMinus, d::Real, upper::Nothing) = true
is_unblocked_above(x::Real, pm::PlusOrMinus, d::Real, upper::Real) =
    (x < upper) | is_negative(pm, d)

is_positive(         x::Real) = x > zero(x)
is_positive(::Plus,  x::Real) = is_positive(x)
is_positive(::Minus, x::Real) = is_negative(x)

is_negative(         x::Real) = x < zero(x)
is_negative(::Plus,  x::Real) = is_negative(x)
is_negative(::Minus, x::Real) = is_positive(x)

# NOTE: The following code relies on IEEE arithmetics to compute the
#       line-search step limits. In particular, it is assumed that comparisons
#       can only be true if none of the operands is NaN. Do not change this
#       code unless you know what you do.
#
# step_to_bound(x,±,d,b) yields the value of the step α to reach the bound b,
# may yield NaN, infinite, or negative value.
step_to_bound(x::T, ::Plus,  d::T, b::T) where {T<:Real} = (b - x)/d
step_to_bound(x::T, ::Minus, d::T, b::T) where {T<:Real} = (x - b)/d
#
# Initialize, update, and finalize αmin.
initial_min_step(::Type{T}) where {T<:AbstractFloat} = typemax(T)
update_min_step(αmin::T, α::T) where {T<:AbstractFloat} = zero(α) ≤ α < αmin ? α : αmin
final_min_step(αmin::AbstractFloat) = αmin
#
# Initialize, update, and finalize αmax.
initial_max_step(::Type{T}) where {T<:AbstractFloat} = -one(T)
update_max_step(αmax::T, α::T) where {T<:AbstractFloat} = α > αmax ? α : αmax
final_max_step(αmax::AbstractFloat) = αmax ≥ zero(αmax) ? αmax : typemax(αmax)

function update_limits(αmin::T, αmax::T, x::T, pm::PlusOrMinus, d::T,
                       lower::L, upper::U) where {T<:AbstractFloat,
                                                  L<:Union{T,Nothing},
                                                  U<:Union{T,Nothing}}
    if L === T
        α = step_to_bound(x, pm, d, lower)
        αmin = update_min_step(αmin, α)
        αmax = update_max_step(αmax, α)
    end
    if U === T
        α = step_to_bound(x, pm, d, upper)
        αmin = update_min_step(αmin, α)
        αmax = update_max_step(αmax, α)
    end
    return αmin, αmax
end

function update_min_step(αmin::T, x::T, pm::PlusOrMinus, d::T,
                         lower::L, upper::U) where {T<:AbstractFloat,
                                                    L<:Union{T,Nothing},
                                                    U<:Union{T,Nothing}}
    if L === T
        α = step_to_bound(x, pm, d, lower)
        αmin = update_min_step(αmin, α)
    end
    if U === T
        α = step_to_bound(x, pm, d, upper)
        αmin = update_min_step(αmin, α)
    end
    return αmin
end

function update_max_step(αmax::T, x::T, pm::PlusOrMinus, d::T,
                         lower::L, upper::U) where {T<:AbstractFloat,
                                                    L<:Union{T,Nothing},
                                                    U<:Union{T,Nothing}}
    if L === T
        α = step_to_bound(x, pm, d, lower)
        αmax = update_max_step(αmax, α)
    end
    if U === T
        α = step_to_bound(x, pm, d, upper)
        αmax = update_max_step(αmax, α)
    end
    return αmax
end
