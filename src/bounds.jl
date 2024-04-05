# BoundSet constructors.
function BoundedSet(lower::AbstractArray{<:Any,N}, upper::AbstractArray{<:Any,N}) where {N}
    T = float(promote_type(eltype(lower), eltype(upper)))
    return BoundedSet{T}(lower, upper)
end

function BoundedSet{T}(lower::AbstractArray{<:Any,N}, upper::AbstractArray{<:Any,N}) where {T,N}
    return BoundedSet{T}(convert_eltype(T, lower), convert_eltype(T, upper))
end

function BoundedSet{T,N}(lower::AbstractArray{<:Any,N}, upper::AbstractArray{<:Any,N}) where {T,N}
    return BoundedSet{T}(lower, upper)
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
BoundedSet{T,N}(Ω::BoundedSet{<:Any,N}) where {T,N} = BoundedSet{T}(Ω)

Base.convert(::Type{BoundedSet}, Ω::BoundedSet) = BoundedSet(Ω)
Base.convert(::Type{BoundedSet{T}}, Ω::BoundedSet) where {T} = BoundedSet{T}(Ω)
Base.convert(::Type{BoundedSet{T,N}}, Ω::BoundedSet) where {T,N} = BoundedSet{T,N}(Ω)
Base.convert(::Type{BoundedSet{T,N,L}}, Ω::BoundedSet) where {T,N,L<:AbstractArray{T,N}} =
    BoundedSet{T,N}(convert(L, Ω.lower), Ω.upper)
Base.convert(::Type{BoundedSet{T,N,L,U}}, Ω::BoundedSet) where {T,N,L<:AbstractArray{T,N},U<:AbstractArray{T,N}} =
    BoundedSet{T,N}(convert(L, Ω.lower), convert(U, Ω.upper))

Base.eltype(Ω::BoundedSet) = eltype(typeof(Ω))
Base.eltype(::Type{<:BoundedSet{T,N}}) where {T,N} = AbstractArray{T,N}

Base.length(Ω::BoundedSet) = length(typeof(Ω))
Base.length(::Type{<:BoundedSet}) = 2 # 1=>lower, 2=>upper

Base.IteratorSize(Ω::BoundedSet) = Base.IteratorSize(typeof(Ω))
Base.IteratorSize(::Type{<:BoundedSet}) = Base.HasLength()

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
feasible_bounds(lower::AbstractArray, upper::Nothing) = !any(isnan, lower)
feasible_bounds(lower::Nothing, upper::AbstractArray) = !any(isnan, upper)
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

# Scalar methods to project a variable.
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

# Scalar method to project a search direction (see notes about
# `project_variable`).
project_direction(x, pm::PlusOrMinus, d, lower, upper) =
    ifelse(can_vary(x, pm, d, lower, upper), d, zero(d))

"""
    changing_variables!([E,] dst, x, ±, d, Ω) -> dst

overwrites destination `dst` with ones where variables in `x` will vary along
direction `±d` within the bounds implemented by `Ω` and zeros elsewhere.

For efficiency, it is assumed without checking that `x ∈ Ω` holds.

Optional argument `E` specifies which *engine* to use for the computations. If
unspecified, `E = NumOptBase.engine(dst, x, d, Ω...)` is assumed.

Note that `x[i]` is non-changing if `d[i] = 0`. Hence, if `±d = -∇f(x)`, with
`∇f(x)` the gradient of the objective function, the destination is set to zero
everywhere the Karush-Kuhn-Tucker (K.K.T.) conditions are satisfied for the
problem:

    minₓ f(x) s.t. x ∈ Ω

In other words, `all(izero, dst)` holds for (exact) convergence.

"""
function changing_variables!(dst::AbstractArray{<:Real,N},
                             x::AbstractArray{T,N},
                             pm::PlusOrMinus,
                             d::AbstractArray{T,N},
                             Ω::BoundedSet{T,N}) where {T,N}
    return changing_variables!(engine(dst, x, d, Ω...), dst, x, pm, d, Ω)
end

function changing_variables!(::Type{E},
                             dst::AbstractArray{<:Real,N},
                             x::AbstractArray{T,N},
                             pm::PlusOrMinus,
                             d::AbstractArray{T,N},
                             Ω::BoundedSet{T,N}) where {E<:Engine,T,N}
    # NOTE: See comments in `project_variables!`.
    check_axes(x; dest=dst, dir=d, lower=Ω.lower, upper=Ω.upper)
    below, above = is_bounding(Ω)
    if below & above
        unsafe_changing_variables!(E, dst, x, pm, d, Ω.lower, Ω.upper)
    elseif below
        unsafe_changing_variables!(E, dst, x, pm, d, Ω.lower, nothing)
    elseif above
        unsafe_changing_variables!(E, dst, x, pm, d, nothing, Ω.upper)
    else
        fill!(dst, one(eltype(dst)))
    end
    return dst
end

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
        function unsafe_changing_variables!(::Type{<:$engine},
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
    end
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

# For `linesearch_limits`, use the same trick as for the `extrema` function in
# `base/reduce.jl`.
struct Twice{F} <: Function
    func::F
end
@inline (obj::Twice)(args...; kwds...) = twice(obj.func(args...; kwds...))
@inline twice(x) = (x, x)

for what in (:limits, :stepmin, :stepmax)
    func = Symbol("linesearch_",what)
    unsafe_func = Symbol("unsafe_",func)
    initial = Symbol(what,"_initial")
    reduce = Symbol(what,"_reduce")
    final = Symbol(what,"_final")
    filter = what === :limits ? :Twice : :identity
    rtype = what === :limits ? :(Tuple{T,T}) : :(T)
    @eval begin
        function $func(x::AbstractArray{T,N},
                       pm::PlusOrMinus, d::AbstractArray{T,N},
                       Ω::BoundedSet{T,N}) where {T,N}
            return $func(engine(x, d, Ω...), x, pm, d, Ω)
        end

        function $func(::Type{E}, x::AbstractArray{T,N},
                       pm::PlusOrMinus, d::AbstractArray{T,N},
                       Ω::BoundedSet{T,N}) where {E<:Engine,T,N}
            # NOTE: See comments in `project_variables!`.
            check_axes(x; dir=d, lower=Ω.lower, upper=Ω.upper)
            return $unsafe_func(E, x, pm, d, Ω...)
        end
        function $unsafe_func(::Type{E}, x::AbstractArray{T,N},
                              pm::PlusOrMinus, d::AbstractArray{T,N},
                              l::AbstractArray{T,N},
                              u::AbstractArray{T,N}) where {E<:Engine,T,N}
            below, above = is_bounding(l, u)
            r = $initial(T)::$(rtype)
            if below & above
                r = mapreduce($filter(step_to_bounds(     pm)), $reduce, x, d, l, u; init=r)::$(rtype)
            elseif below
                r = mapreduce($filter(step_to_lower_bound(pm)), $reduce, x, d, l   ; init=r)::$(rtype)
            elseif above
                r = mapreduce($filter(step_to_upper_bound(pm)), $reduce, x, d,    u; init=r)::$(rtype)
            end
            return $final(r)::$(rtype)
        end
    end
end

@inline stepmin_initial(::Type{T}) where {T} = typemax(T)
@inline stepmax_initial(::Type{T}) where {T} = typemin(T)
@inline limits_initial(::Type{T}) where {T} = (stepmin_initial(T), stepmax_initial(T))

@inline stepmin_final(αₘᵢₙ) = αₘᵢₙ
@inline stepmax_final(αₘₐₓ) = αₘₐₓ ≥ zero(αₘₐₓ) ? αₘₐₓ : typemax(αₘₐₓ)
@inline limits_final((αₘᵢₙ, αₘₐₓ)) = (stepmin_final(αₘᵢₙ), stepmax_final(αₘₐₓ))

# The following functions yields the min./max. step size between `a` and `b`.
#
# NOTE: These functions are intended to be used as the reduce operator in a
#       call to `mapreduce`. They do not assume a specific order of their
#       arguments but rely on (i) the initial value to not be a NaN and (i)
#       IEEE rules that a comparison involving a NaN always yields false.
@inline stepmin_reduce(a::T, b::T) where {T} = (isnan(b) | (a < b)) ? a : b
@inline stepmax_reduce(a::T, b::T) where {T} = (isnan(b) | (a > b)) ? a : b
@inline limits_reduce((min1, max1), (min2, max2)) = (stepmin_reduce(min1, min2),
                                                     stepmax_reduce(max1, max2))


"""
    NumOptBase.step_choice(d, α₋, α₊, β = bad_step(T)) -> α

chooses the step to a bound according to the sign of `d`. The result is `α₋` if
`d < 0`, `α₊` if `d > 0`, `β` otherwise (`d` is zero or NaN).

"""
@inline step_choice(d::T, neg::T, pos::T, bad::T = bad_step(T)) where {T} =
    ifelse(d < zero(d), neg, ifelse(d > zero(d), pos, bad))
"""
    NumOptBase.step_to_bounds(±) -> f

yields the function `f` such that `f(x,d,l,u)` yields the same result as
`step_to_bounds(x,±,d,l,u)`.

"""
step_to_bounds(::Plus) = step_to_bounds
step_to_bounds(::Minus) = step_from_bounds

"""
    NumOptBase.step_to_lower_bound(±) -> f

yields the function `f` such that `f(x,d,l)` yields the same result as
`step_to_lower_bound(x,±,d,l)`.

"""
step_to_lower_bound(::Plus) = step_to_lower_bound
step_to_lower_bound(::Minus) = step_from_lower_bound

"""
    NumOptBase.step_to_upper_bound(±) -> f

yields the function `f` such that `f(x,d,u)` yields the same result as
`step_to_upper_bound(x,±,d,u)`.

"""
step_to_upper_bound(::Plus) = step_to_upper_bound
step_to_upper_bound(::Minus) = step_from_upper_bound

"""
    NumOptBase.step_to_bounds(x, d, l, u) -> α
    NumOptBase.step_to_bounds(x, +, d, l, u) -> α

for the variable `x` such that `l ≤ x ≤ u` and search direction `d`, yield the
nonnegative step `α` to one of the bounds `l` or `u` depending on the sign of
`d`:

- if `d < 0`, `α ≥ 0` such that `x + α*d = l` is returned;
- if `d > 0`, `α ≥ 0` such that `x + α*d = u` is returned;
- otherwise, `α = NaN` is returned.

"""
@inline step_to_bounds(x::T, d::T, l::T, u::T) where {T} =
    step_choice(d, (l - x)/d, (u - x)/d)

"""
    NumOptBase.step_from_bounds(x, d, l, u) -> α
    NumOptBase.step_to_bounds(x, -, d, l, u) -> α

for the variable `x` such that `l ≤ x ≤ u` and search direction `d`, yield the
nonnegative step `α` from one of the bounds `l` or `u` depending on the sign of
`d`:

- if `d < 0`, `α ≥ 0` such that `x - α*d = u` is returned;
- if `d > 0`, `α ≥ 0` such that `x - α*d = l` is returned;
- otherwise, `α = NaN` is returned.

"""
@inline step_from_bounds(x::T, d::T, l::T, u::T) where {T} =
    step_choice(d, (x - u)/d, (x - l)/d)

"""
    NumOptBase.step_to_lower_bound(x, d, l) -> α
    NumOptBase.step_to_lower_bound(x, +, d, l) -> α

for the variable `x` such that `l ≤ x` and search direction `d`, yield `+Inf`,
`NaN`, or the nonnegative step `α` to the lower bound `l` depending on the
sign of `d`:

- if `d < 0`, `α ≥ 0` such that `x + α*d = l` is returned;
- if `d > 0`, `α = +Inf` is returned;
- otherwise, `α = NaN` is returned.

"""
@inline step_to_lower_bound(x::T, d::T, l::T) where {T} =
    step_choice(d, (l - x)/d, typemax(T))

"""
    NumOptBase.step_from_lower_bound(x, d, l) -> α
    NumOptBase.step_to_lower_bound(x, -, d, l) -> α

for the variable `x` such that `l ≤ x` and search direction `d`, yield `+Inf`,
`NaN`, or the nonnegative step `α` from the lower bound `l` depending on the
sign of `d`:

- if `d < 0`, `α = +Inf` is returned;
- if `d > 0`, `α ≥ 0` such that `x - α*d = l` is returned;
- otherwise, `α = NaN` is returned.

"""
@inline step_from_lower_bound(x::T, d::T, l::T) where {T} =
    step_choice(d, typemax(T), (x - l)/d)

"""
    NumOptBase.step_to_upper_bound(x, d, u) -> α
    NumOptBase.step_to_upper_bound(x, +, d, u) -> α

for the variable `x` such that `x ≤ u` and search direction `d`, yield `+Inf`,
`NaN`, or the nonnegative step `α` to the upper bound `u` depending on the
sign of `d`:

- if `d < 0`, `α = +Inf` is returned;
- if `d > 0`, `α ≥ 0` such that `x + α*d = u` is returned;
- otherwise, `α = NaN` is returned.

"""
@inline step_to_upper_bound(x::T, d::T, u::T) where {T} =
    step_choice(d, typemax(T), (u - x)/d)

"""
    NumOptBase.step_from_upper_bound(x, d, u) -> α
    NumOptBase.step_to_upper_bound(x, -, d, u) -> α

for the variable `x` such that `x ≤ u` and search direction `d`, yield `+Inf`,
`NaN`, or the nonnegative step `α` from the upper bound `u` depending on the
sign of `d`:

- if `d < 0`, `α ≥ 0` such that `x - α*d = u` is returned;
- if `d > 0`, `α = +Inf` is returned;
- otherwise, `α = NaN` is returned.

"""
@inline step_from_upper_bound(x::T, d::T, u::T) where {T} =
    step_choice(d, (x - u)/d, typemax(T))

step_to_bounds(     x, ::Plus,  d, l, u) = step_to_bounds(     x, d, l, u)
step_to_lower_bound(x, ::Plus,  d, l   ) = step_to_lower_bound(x, d, l   )
step_to_upper_bound(x, ::Plus,  d,    u) = step_to_upper_bound(x, d,    u)

step_to_bounds(     x, ::Minus, d, l, u) = step_from_bounds(     x, d, l, u)
step_to_lower_bound(x, ::Minus, d, l   ) = step_from_lower_bound(x, d, l   )
step_to_upper_bound(x, ::Minus, d,    u) = step_from_upper_bound(x, d,    u)

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
