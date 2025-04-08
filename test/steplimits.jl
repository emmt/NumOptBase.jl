"""

Module `TestingLineSearchLimits` is to determine the best implementation of
`linesearch_limits`, `linesearch_stepmin`, and `linesearch_stepmax`.

Lessons learned:

- use `@inbounds`;

- *avoid branching* (use `ifselse` for alternatives);

- manage to fix expressions so that `@fastmath` can be used;

- use *fast reductions*, like `smin = (s < smin ? s : smin)`;

- with `@inbounds @fastmath`, `@simd` is not necessary (automatically done by the
  optimizer);

- avoid predictable operations like negating `d[i]`;

- the `mapreduce` versions are not the slowest but they allocates temporaries,
  they are faster without the `init` keyword, they may be used with GPU arrays;

"""
module TestingLineSearchLimits

using LoopVectorization
using BenchmarkTools
using ArrayTools
using TypeUtils
using StructuredArrays
using StructuredArrays: value
const PlusOrMinus = Union{typeof(+),typeof(-)}

"""
    linesearch_limits(Val(impl), x, pm, d, lo, hi) -> smin, smax

Determine the limits `smin` and `smax` for the step length `s` in a line-search where
iterates `x` are given by:

    xp = proj(x ± s*d)

where `proj(x)` denotes the orthogonal projection on the convex set defined by separable
lower and upper bounds `lo` and `hi` (unless `nothing`) and where `±` is specified
by `pm` which is one of `+` or `-`.

On return, `smin` is the largest nonnegative step length such that if `0 ≤ s ≤ smin`,
then:

    proj(x ± s*d) = x ± s*d

On return, `smax` is set to the least nonnegative step length such that if `s ≥ smax ≥ 0`,
then:

    proj(x ± s*d) = proj(x ± smax*d)

In other words, no bounds are overcome if `0 ≤ s ≤ smin` and the projected variables are
all the same for any `s` such that `s ≥ smax ≥ 0`.

The following restrictions and assumptions are not checked (for speed) but must hold for
the code to yield correct results:

Assumptions (`∀i`):

- none of `x[i]`, `d[i]`, `lo[i]`, or `hi[i]` is a NaN;

- `lo[i] ≤ hi[i]`, `lo[i] ∈ ℝ ∪ {-∞}` and `hi[i] ∈ ℝ ∪ {+∞}`; hence, `lo[i]` may be `-∞`
  but not `+∞` and `hi[i]` may be `+∞` but not `-∞`;

- `x` is feasible: `lo[i] ≤ x[i] ≤ hi[i]` hold `∀i`;

- `x[i] ∈ ℝ` and `d[i] ∈ ℝ` are both finite;

Possibilities:

- `x[i]` may be at its lower bound, strictly inside its, bounds, or at is upper bound;
- `d[i]` may be negative, zero, or positive;
- `lo` may be: (i) `-∞` everywhere and quickly inferable as unbounded below or not, (ii)
 `-∞` somewhere and finite elsewhere;

Testing implementations:

- all possible cases must be tested for correctness;

Naming conventions for the implementation:

- `:dot` is implemented with temporary arrays and element-wise operators like `.+` etc.;

- `:ref` is the reference implementation;

- `:gpu` is the GPU implementation;

- `:mapreduce` is the implementation by `mapreduce` method;

- `:vmapreduce` is the implementation by `vmapreduce` method;

- `:turbo` is the `@turbo` implementation (if possible);

- other implementations are symbolic names consisting in:

  - `negate_` prefix means that `pm(d[i])` may be called to negate `d[i]`;

  - `_branching` suffix means that branching is not avoided (this includes `continue` and
    `break` statements); otherwise, there is no branching and `ifelse` (or similar) is
    used to deal with alternatives;

  - `_fastreduction` suffix means that the reductions are done by simple statements like
    `min` or `max`;

  - `inbounds` means that the loop is decorated by `@inbounds`;

  - `inbounds_fastmath` means that the loop is decorated by `@inbounds @fastmath`;

  - `inbounds_simd` means that the loop is decorated by `@inbounds @simd`;

  - `inbounds_fastmath_simd` means that the loop is decorated by `@inbounds @fastmath
    @simd`;

""" linesearch_limits

for f in (:linesearch_limits, :linesearch_stepmin, :linesearch_stepmax)
    @eval function $f(impl::Val, x::AbstractArray{T,N},
                      pm::PlusOrMinus, d::AbstractArray{T,N},
                      lo::Union{AbstractArray{T},T,Nothing},
                      hi::Union{AbstractArray{T},T,Nothing}) where {T,N}
        return $f(impl, x, pm, d,
                  as_lower_bound(x, lo),
                  as_upper_bound(x, hi))
    end
end

function (alg::Val, x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            lo::Union{AbstractArray{T},T,Nothing},
                            hi::Union{AbstractArray{T},T,Nothing}) where {T,N}
    return linesearch_limits(alg, x, pm, d,
                             as_lower_bound(x, lo),
                             as_upper_bound(x, hi))
end

function linesearch_limits(::Val{:dot}, x::AbstractArray{T,N},
                           pm::PlusOrMinus, d::AbstractArray{T,N},
                           lo::AbstractArray{T,N},
                           hi::AbstractArray{T,N}) where {T,N}
    S = float(bare_type(T)) # type of step length
    bounded_below = is_bounded_below_by(lo)
    bounded_above = is_bounded_above_by(hi)
    if !(bounded_below|bounded_above)
        # Quick return if unconstrained.
        return typemax(S), typemax(S)
    end
    forward = pm isa typeof(+) # search forward?
    if any(<(zero(eltype(d))), d) # some d[i] are negative?
        if forward ? bounded_below : bounded_above
            n = forward ? lo .- x : x .- hi
            i = d .< zero(eltype(d))
            s = n[i] ./ d[i]
            smin, smax = extrema(s)::Tuple{S,S}
        else
            smin = typemax(S)
            smax = typemax(S)
        end
    else
        smin = typemax(S)
        smax = typemin(S)
    end
    if any(>(zero(eltype(d))), d) # some d[i] are positive?
        if forward ? bounded_above : bounded_below
            n = forward ? hi .- x : x .- lo
            i = d .> zero(eltype(d))
            s = n[i] ./ d[i]
            if smax < typemax(S)
                smin1, smax1 = extrema(s)::Tuple{S,S}
                smax = max(smax, smax1)
                smin = min(smin, smin1)
            else
                smin = min(smin, minimum(s)::S)
            end
        else
            smax = typemax(S)
        end
    end
    return final_stepmin(smin), final_stepmax(smax)
end

is_bounded_below_by(lo::Nothing) = false
is_bounded_below_by(lo::Number) = lo > typemin(lo)
is_bounded_below_by(lo::AbstractUniformArray) = is_bounded_below_by(value(lo))
is_bounded_below_by(lo::AbstractArray) = true

is_bounded_above_by(hi::Nothing) = false
is_bounded_above_by(hi::Number) = hi < typemax(hi)
is_bounded_above_by(hi::AbstractUniformArray) = is_bounded_above_by(value(hi))
is_bounded_above_by(hi::AbstractArray) = true

as_lower_bound(x::AbstractArray{T,N}, lo::AbstractArray{T,N}) where {T,N} = lo
as_lower_bound(x::AbstractArray{T,N}, lo::Nothing) where {T,N} =
    as_lower_bound(x, typemin(T))
as_lower_bound(x::AbstractArray{T,N}, lo::Number) where {T,N} =
    as_lower_bound(x, UniformArray{T,N}(lo, axes(x)))

as_upper_bound(x::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {T,N} = hi
as_upper_bound(x::AbstractArray{T,N}, hi::Nothing) where {T,N} =
    as_upper_bound(x, typemin(T))
as_upper_bound(x::AbstractArray{T,N}, hi::Number) where {T,N} =
    as_upper_bound(x, UniformArray{T,N}(hi, axes(x)))

# NOTE:
#
# With `@fastmath`, IEEE rules for comparisons involving NaNs are not followed, `iszero`
# is however correct. See
# https://stackoverflow.com/questions/7420665/what-does-gccs-ffast-math-actually-do or
# https://simonbyrne.github.io/notes/fastmath/ which also discusses the case of Julia
#
# Therefore, with `@fastmath`, we use `iszero(d[i])` because `d[i] > zero(d[i])` and `d[i]
# < zero(d[i])` may be wrong.

# NOTE: Type assertion `::S` with `S` the inferred step type is to crash instead of
#       slowing down if the inference is wrong.
#

function linesearch_limits(::Val{impl}, x::AbstractArray{T,N},
                           pm::PlusOrMinus, d::AbstractArray{T,N},
                           lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {impl,T,N}
    @assert_same_axes x d lo hi
    S = step_type(T)
    smin = initial_stepmin(S)
    smax = initial_stepmax(S)
    if impl === :ref
        # Reference version.
        for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            if t < zero(t)
                s = ((lo[i] - x[i])/t)::S
            elseif t > zero(t)
                s = ((hi[i] - x[i])/t)::S
            else
                continue
            end
            smin = (s < smin ? s : smin)
            smax = (s > smax ? s : smax)
        end
    elseif impl === :negate_inbounds_branching
        @inbounds for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            if t < zero(t)
                s = ((lo[i] - x[i])/t)::S
            elseif t > zero(t)
                s = ((hi[i] - x[i])/t)::S
            else
                continue
            end
            smin = (s < smin ? s : smin)
            smax = (s > smax ? s : smax)
        end
    elseif impl === :negate_inbounds_fastmath_branching
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            iszero(t) && continue
            s = (((t < zero(t) ? lo[i] : hi[i]) - x[i])/t)::S
            smin = (s < smin ? s : smin)
            smax = (s > smax ? s : smax)
        end
    elseif impl === :negate_inbounds
        @inbounds for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            s = forward_step_to_bound(x[i], t, lo[i], hi[i])::S
            nz = !iszero(t)
            smin = (nz & (s < smin) ? s : smin)
            smax = (nz & (s > smax) ? s : smax)
        end
    elseif impl === :negate_inbounds_fastmath
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            s = forward_step_to_bound(x[i], t, lo[i], hi[i])::S
            nz = !iszero(t)
            smin = (nz & (s < smin) ? s : smin)
            smax = (nz & (s > smax) ? s : smax)
        end
    elseif impl === :negate_inbounds_fastreduction
        @inbounds for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            s = forward_step_to_bound(x[i], t, lo[i], hi[i])::S
            z = iszero(t)
            smn = ifelse(z, typemax(s), s)
            smx = ifelse(z, typemin(s), s)
            smin = (smn < smin ? smn : smin)
            smax = (smx > smax ? smx : smax)
        end
    elseif impl === :negate_inbounds_fastmath_fastreduction
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            s = forward_step_to_bound(x[i], t, lo[i], hi[i])::S
            z = iszero(t)
            smn = ifelse(z, typemax(s), s)
            smx = ifelse(z, typemin(s), s)
            smin = (smn < smin ? smn : smin)
            smax = (smx > smax ? smx : smax)
        end
    elseif impl === :negate_inbounds_simd_fastreduction
        @inbounds @simd for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            s = forward_step_to_bound(x[i], t, lo[i], hi[i])::S
            z = iszero(t)
            smn = ifelse(z, typemax(s), s)
            smx = ifelse(z, typemin(s), s)
            smin = (smn < smin ? smn : smin)
            smax = (smx > smax ? smx : smax)
        end
    elseif impl === :negate_inbounds_fastmath_simd_fastreduction
        @inbounds @fastmath @simd for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            s = forward_step_to_bound(x[i], t, lo[i], hi[i])::S
            z = iszero(t)
            smn = ifelse(z, typemax(s), s)
            smx = ifelse(z, typemin(s), s)
            smin = (smn < smin ? smn : smin)
            smax = (smx > smax ? smx : smax)
        end
    elseif impl === :inbounds_branching
        @inbounds for i in eachindex(x, d, lo, hi)
            t = d[i]
            iszero(t) && continue
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
            smax = (s > smax ? s : smax)
        end
    elseif impl === :inbounds_fastmath_branching
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = d[i]
            iszero(t) && continue
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
            smax = (s > smax ? s : smax)
        end
    elseif impl === :inbounds
        @inbounds for i in eachindex(x, d, lo, hi)
            t = d[i]
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            nz = !iszero(t)
            smin = (nz & (s < smin) ? s : smin)
            smax = (nz & (s > smax) ? s : smax)
        end
    elseif impl === :inbounds_fastmath
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = d[i]
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            nz = !iszero(t)
            smin = (nz & (s < smin) ? s : smin)
            smax = (nz & (s > smax) ? s : smax)
        end
    elseif impl === :inbounds_fastreduction
        @inbounds for i in eachindex(x, d, lo, hi)
            t = d[i]
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            z = iszero(t)
            smn = ifelse(z, typemax(s), s)
            smx = ifelse(z, typemin(s), s)
            smin = (smn < smin ? smn : smin)
            smax = (smx > smax ? smx : smax)
        end
    elseif impl === :inbounds_fastmath_fastreduction
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = d[i]
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            z = iszero(t)
            smn = ifelse(z, typemax(s), s)
            smx = ifelse(z, typemin(s), s)
            smin = (smn < smin ? smn : smin)
            smax = (smx > smax ? smx : smax)
        end
    elseif impl === :inbounds_simd_fastreduction
        @inbounds @simd for i in eachindex(x, d, lo, hi)
            t = d[i]
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            z = iszero(t)
            smn = ifelse(z, typemax(s), s)
            smx = ifelse(z, typemin(s), s)
            smin = (smn < smin ? smn : smin)
            smax = (smx > smax ? smx : smax)
        end
    elseif impl === :inbounds_fastmath_simd_fastreduction
        @inbounds @fastmath @simd for i in eachindex(x, d, lo, hi)
            t = d[i]
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            z = iszero(t)
            smn = ifelse(z, typemax(s), s)
            smx = ifelse(z, typemin(s), s)
            smin = (smn < smin ? smn : smin)
            smax = (smx > smax ? smx : smax)
        end
    else
        smin = linesearch_stepmin(Val(impl), x, pm, d, lo, hi)::S
        smax = linesearch_stepmax(Val(impl), x, pm, d, lo, hi)::S
    end
    return final_stepmin(smin), final_stepmax(smax)
end

function linesearch_stepmin(::Val{impl}, x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {impl,T,N}
    @assert_same_axes x d lo hi
    S = step_type(T)
    smin = initial_stepmin(S)
    if impl === :ref
        for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            if t < zero(t)
                s = ((lo[i] - x[i])/t)::S
            elseif t > zero(t)
                s = ((hi[i] - x[i])/t)::S
            else
                continue
            end
            smin = (s < smin ? s : smin)
        end
    elseif impl === :negate_inbounds_branching
        @inbounds for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            if t < zero(t)
                s = ((lo[i] - x[i])/t)::S
            elseif t > zero(t)
                s = ((hi[i] - x[i])/t)::S
            else
                continue
            end
            smin = (s < smin ? s : smin)
        end
    elseif impl === :negate_inbounds_fastmath_branching
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            iszero(t) && continue
            s = (((t < zero(t) ? lo[i] : hi[i]) - x[i])/t)::S
            smin = (s < smin ? s : smin)
        end
    elseif impl === :negate_inbounds
        @inbounds for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            s = forward_step_to_bound(x[i], t, lo[i], hi[i])::S
            smin = ((!iszero(t)) & (s < smin) ? s : smin)
        end
    elseif impl === :negate_inbounds_fastmath
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            s = forward_step_to_bound(x[i], t, lo[i], hi[i])::S
            smin = ((!iszero(t)) & (s < smin) ? s : smin)
        end
    elseif impl === :negate_inbounds_fastreduction
        @inbounds for i in eachindex(x, d, lo, hi)
            s = forward_stepmin_to_bound(x[i], pm(d[i]), lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
        end
    elseif impl === :negate_inbounds_fastmath_fastreduction
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            s = forward_stepmin_to_bound(x[i], pm(d[i]), lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
        end
    elseif impl === :negate_inbounds_simd_fastreduction
        @inbounds @simd for i in eachindex(x, d, lo, hi)
            s = forward_stepmin_to_bound(x[i], pm(d[i]), lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
        end
    elseif impl === :negate_inbounds_fastmath_simd_fastreduction
        @inbounds @fastmath @simd for i in eachindex(x, d, lo, hi)
            s = forward_stepmin_to_bound(x[i], pm(d[i]), lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
        end
    elseif impl === :inbounds_branching
        @inbounds for i in eachindex(x, d, lo, hi)
            t = d[i]
            iszero(t) && continue
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
        end
    elseif impl === :inbounds_fastmath_branching
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = d[i]
            iszero(t) && continue
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
        end
    elseif impl === :inbounds
        @inbounds for i in eachindex(x, d, lo, hi)
            t = d[i]
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            smin = ((!iszero(t)) & (s < smin) ? s : smin)
        end
    elseif impl === :inbounds_fastmath
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = d[i]
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            smin = ((!iszero(t)) & (s < smin) ? s : smin)
        end
    elseif impl === :inbounds_fastreduction
        @inbounds for i in eachindex(x, d, lo, hi)
            s = stepmin_to_bound(x[i], pm, d[i], lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
        end
    elseif impl === :inbounds_fastmath_fastreduction
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            s = stepmin_to_bound(x[i], pm, d[i], lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
        end
    elseif impl === :inbounds_simd_fastreduction
        @inbounds @simd for i in eachindex(x, d, lo, hi)
            s = stepmin_to_bound(x[i], pm, d[i], lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
        end
    elseif impl === :inbounds_fastmath_simd_fastreduction
        @inbounds @fastmath @simd for i in eachindex(x, d, lo, hi)
            s = stepmin_to_bound(x[i], pm, d[i], lo[i], hi[i])::S
            smin = (s < smin ? s : smin)
        end
    else
        error("unknown implementation `$(repr(impl))`")
    end
    return final_stepmin(smin)
end

function linesearch_stepmax(::Val{impl}, x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {impl,T,N}
    @assert_same_axes x d lo hi
    S = step_type(T)
    smax = initial_stepmax(S)
    if impl === :ref
        for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            if t < zero(t)
                s = ((lo[i] - x[i])/t)::S
            elseif t > zero(t)
                s = ((hi[i] - x[i])/t)::S
            else
                continue
            end
            smax = (s > smax ? s : smax)
        end
    elseif impl === :negate_inbounds_branching
        @inbounds for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            if t < zero(t)
                s = ((lo[i] - x[i])/t)::S
            elseif t > zero(t)
                s = ((hi[i] - x[i])/t)::S
            else
                continue
            end
            smax = (s > smax ? s : smax)
        end
    elseif impl === :negate_inbounds_fastmath_branching
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            iszero(t) && continue
            s = (((t < zero(t) ? lo[i] : hi[i]) - x[i])/t)::S
            smax = (s > smax ? s : smax)
        end
    elseif impl === :negate_inbounds
        @inbounds for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            s = forward_step_to_bound(x[i], t, lo[i], hi[i])::S
            smax = ((!iszero(t)) & (s > smax) ? s : smax)
        end
    elseif impl === :negate_inbounds_fastmath
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = pm(d[i])
            s = forward_step_to_bound(x[i], t, lo[i], hi[i])::S
            smax = ((!iszero(t)) & (s > smax) ? s : smax)
        end
    elseif impl === :negate_inbounds_fastreduction
        @inbounds for i in eachindex(x, d, lo, hi)
            s = forward_stepmax_to_bound(x[i], pm(d[i]), lo[i], hi[i])::S
            smax = (s > smax ? s : smax)
        end
    elseif impl === :negate_inbounds_fastmath_fastreduction
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            s = forward_stepmax_to_bound(x[i], pm(d[i]), lo[i], hi[i])::S
            smax = (s > smax ? s : smax)
        end
    elseif impl === :negate_inbounds_simd_fastreduction
        @inbounds @simd for i in eachindex(x, d, lo, hi)
            s = forward_stepmax_to_bound(x[i], pm(d[i]), lo[i], hi[i])::S
            smax = (s > smax ? s : smax)
        end
    elseif impl === :negate_inbounds_fastmath_simd_fastreduction
        @inbounds @fastmath @simd for i in eachindex(x, d, lo, hi)
            s = forward_stepmax_to_bound(x[i], pm(d[i]), lo[i], hi[i])::S
            smax = (s > smax ? s : smax)
        end
    elseif impl === :inbounds_branching
        @inbounds for i in eachindex(x, d, lo, hi)
            t = d[i]
            iszero(t) && continue
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            smax = (s > smax ? s : smax)
        end
    elseif impl === :inbounds_fastmath_branching
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = d[i]
            iszero(t) && continue
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            smax = (s > smax ? s : smax)
        end
    elseif impl === :inbounds
        @inbounds for i in eachindex(x, d, lo, hi)
            t = d[i]
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            smax = ((!iszero(t)) & (s > smax) ? s : smax)
        end
    elseif impl === :inbounds_fastmath
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            t = d[i]
            s = step_to_bound(x[i], pm, t, lo[i], hi[i])::S
            smax = ((!iszero(t)) & (s > smax) ? s : smax)
        end
    elseif impl === :inbounds_fastreduction
        @inbounds for i in eachindex(x, d, lo, hi)
            s = stepmax_to_bound(x[i], pm, d[i], lo[i], hi[i])::S
            smax = (s > smax ? s : smax)
        end
    elseif impl === :inbounds_fastmath_fastreduction
        @inbounds @fastmath for i in eachindex(x, d, lo, hi)
            s = stepmax_to_bound(x[i], pm, d[i], lo[i], hi[i])::S
            smax = (s > smax ? s : smax)
        end
    elseif impl === :inbounds_simd_fastreduction
        @inbounds @simd for i in eachindex(x, d, lo, hi)
            s = stepmax_to_bound(x[i], pm, d[i], lo[i], hi[i])::S
            smax = (s > smax ? s : smax)
        end
    elseif impl === :inbounds_fastmath_simd_fastreduction
        @inbounds @fastmath @simd for i in eachindex(x, d, lo, hi)
            s = stepmax_to_bound(x[i], pm, d[i], lo[i], hi[i])::S
            smax = (s > smax ? s : smax)
        end
    else
        error("unknown implementation `$(repr(impl))`")
    end
    return final_stepmax(smax)
end

# Yield step type for variables of element type `T`.
step_type(::Type{T}) where {T} = float(bare_type(T))

initial_stepmin(::Type{T}) where {T<:AbstractFloat} = typemax(T)
initial_stepmax(::Type{T}) where {T<:AbstractFloat} = typemin(T)

final_stepmin(smin::T) where {T<:AbstractFloat} = smin
final_stepmax(smax::T) where {T<:AbstractFloat} = ifelse(smax ≥ zero(T), smax, typemax(T))

# `s = step_to_bound(x,+,d,lo,hi)` or `s = forward_step_to_bound(x,d,lo,hi)` yield the
# step `s` to the bound in the forward direction `+d`. If `d` is zero, the result is
# invalid; hence, `fix_stepmin(d,s)` and/or `fix_stepmax(d,s)` must be called next.
@inline forward_step_to_bound(x, d, lo, hi) = ((d < zero(d) ? lo : hi) - x)/d

# `s = step_to_bound(x,-,d,lo,hi)` or `s = backward_step_to_bound(x,d,lo,hi)` yield the
# step `s` to the bound in the backward direction `-d`. If `d` is zero, the result is
# invalid; hence, `fix_stepmin(d,s)` and/or `fix_stepmax(d,s)` must be called next.
@inline backward_step_to_bound(x, d, lo, hi) = (x - (d > zero(d) ? lo : hi))/d

# `fix_stepmin(d,s)` and `fix_stepmax(d,s)` fix the step `s` for direction `d` for one of
# the line-search step limits. The result is suitable for fast reduction.
@inline fix_stepmin(d, s) = ifelse(iszero(d), typemax(s), s)
@inline fix_stepmax(d, s) = ifelse(iszero(d), typemin(s), s)

for f in (:step_to_bound, :stepmin_to_bound, :stepmax_to_bound)
    @eval begin
        @inline $f(x, ::typeof(+), d, lo, hi) = $(Symbol( "forward_",f))(x, d, lo, hi)
        @inline $f(x, ::typeof(-), d, lo, hi) = $(Symbol("backward_",f))(x, d, lo, hi)
    end
end

for dir in (:forward, :backward), lim in (:min, :max)
    @eval begin
        @inline $(Symbol(dir,"_step",lim,"_to_bound"))(x, d, lo, hi) =
            $(Symbol("fix_step",lim,))(d, $(Symbol(dir,"_step_to_bound"))(x, d, lo, hi))
    end
end

# NOTE Type assertion is not possible in `@turbo` loops.

function linesearch_limits(::Val{:turbo}, x::AbstractArray{T,N},
                           pm::PlusOrMinus, d::AbstractArray{T,N},
                           lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {T,N}
    @assert_same_axes x d lo hi
    S = step_type(T)
    smin = initial_stepmin(S)
    smax = initial_stepmax(S)
    if pm === +
        @turbo for i in eachindex(x, d, lo, hi)
            s = forward_step_to_bound(x[i], d[i], lo[i], hi[i])
            smn = fix_stepmin(t, s)
            smx = fix_stepmax(t, s)
            smin = (smn < smin ? smn : smin)
            smax = (smx > smax ? smx : smax)
        end
    else
        @turbo for i in eachindex(x, d, lo, hi)
            s = backward_step_to_bound(x[i], d[i], lo[i], hi[i])
            smn = fix_stepmin(t, s)
            smx = fix_stepmax(t, s)
            smin = (smn < smin ? smn : smin)
            smax = (smx > smax ? smx : smax)
        end
    end
    return final_stepmin(smin), final_stepmax(smax)
end

function linesearch_stepmin(::Val{:turbo}, x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {T,N}
    @assert_same_axes x d lo hi
    S = step_type(T)
    smin = initial_stepmin(S)
    if pm === +
        @turbo for i in eachindex(x, d, lo, hi)
            s = forward_stepmin_to_bound(x[i], d[i], lo[i], hi[i])
            smin = (s < smin ? s : smin)
        end
    else
        @turbo for i in eachindex(x, d, lo, hi)
            s = backward_stepmin_to_bound(x[i], d[i], lo[i], hi[i])
            smin = (s < smin ? s : smin)
        end
    end
    return final_stepmin(smin)
end

function linesearch_stepmax(::Val{:turbo}, x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {T,N}
    @assert_same_axes x d lo hi
    S = step_type(T)
    smax = initial_stepmax(S)
    if pm === +
        @turbo for i in eachindex(x, d, lo, hi)
            s = forward_stepmax_to_bound(x[i], d[i], lo[i], hi[i])
            smax = (s > smax ? s : smax)
        end
    else
        @turbo for i in eachindex(x, d, lo, hi)
            s = backward_stepmax_to_bound(x[i], d[i], lo[i], hi[i])
            smax = (s > smax ? s : smax)
        end
    end
    return final_stepmax(smax)
end

function linesearch_stepmin(::Val{:vmapreduce}, x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {T,N}
    @assert_same_axes x d lo hi
    f = pm isa typeof(+) ? forward_stepmin_to_bound : backward_stepmin_to_bound
    smin = vmapreduce(f, min, x, d, lo, hi)
    return final_stepmin(smin)
end

function linesearch_stepmax(::Val{:vmapreduce}, x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {T,N}
    @assert_same_axes x d lo hi
    f = pm isa typeof(+) ? forward_stepmax_to_bound : backward_stepmax_to_bound
    smax = vmapreduce(f, min, x, d, lo, hi)
    return final_stepmax(smax)
end

function linesearch_stepmin(::Val{:mapreduce}, x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {T,N}
    @assert_same_axes x d lo hi
    S = step_type(T)
    f = pm isa typeof(+) ? forward_stepmin_to_bound : backward_stepmin_to_bound
    smin = mapreduce(f, min, x, d, lo, hi)::S
    return final_stepmin(smin)
end

function linesearch_stepmax(::Val{:mapreduce}, x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {T,N}
    @assert_same_axes x d lo hi
    S = step_type(T)
    f = pm isa typeof(+) ? forward_stepmax_to_bound : backward_stepmax_to_bound
    smax = mapreduce(f, max, x, d, lo, hi)::S
    return final_stepmax(smax)
end

function linesearch_stepmin(::Val{:mapreduce_init}, x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {T,N}
    @assert_same_axes x d lo hi
    S = step_type(T)
    f = pm isa typeof(+) ? forward_stepmin_to_bound : backward_stepmin_to_bound
    smin = mapreduce(f, min, x, d, lo, hi; init=initial_stepmin(S))::S
    return final_stepmin(smin)
end

function linesearch_stepmax(::Val{:mapreduce_init}, x::AbstractArray{T,N},
                            pm::PlusOrMinus, d::AbstractArray{T,N},
                            lo::AbstractArray{T,N}, hi::AbstractArray{T,N}) where {T,N}
    @assert_same_axes x d lo hi
    S = step_type(T)
    f = pm isa typeof(+) ? forward_stepmax_to_bound : backward_stepmax_to_bound
    smax = mapreduce(f, max, x, d, lo, hi; init=initial_stepmax(S))::S
    return final_stepmax(smax)
end

function runtests(::Type{T} = Float32; dims=(10_000,), kwds...) where {T}
    d = rand(T, dims) .- one(T)/2
    d[rand(Float64, dims) .< 0.05] .= zero(T) # 5% of zeros
    lo = rand(T, dims) .- one(T)/5
    hi = rand(T, dims) .+ lo
    x = clamp.(rand(T, dims) .- one(T)/2, lo, hi)
    runtests(x, d, lo, hi; kwds...)
end

function runtests(x::AbstractArray, d::AbstractArray, lo=nothing, hi=nothing;
                  bench::Bool=true, what = :limits)
    nops = (what === :limits ? 6 : 5)*length(x)
    print("Testing `linesearch_$what` with T=$(eltype(x)), dims=$(repr(size(x))) --> nops = $nops\n")
    print("- forward direction:\n")
    impls = (:dot, :ref, :mapreduce, :mapreduce_init,
             :negate_inbounds_branching,
             :negate_inbounds_fastmath_branching,
             :negate_inbounds,
             :negate_inbounds_fastmath,
             :negate_inbounds_fastreduction,
             :negate_inbounds_fastmath_fastreduction,
             :negate_inbounds_simd_fastreduction,
             :negate_inbounds_fastmath_simd_fastreduction,
             :inbounds_branching,
             :inbounds_fastmath_branching,
             :inbounds,
             :inbounds_fastmath,
             :inbounds_fastreduction,
             :inbounds_fastmath_fastreduction,
             :inbounds_simd_fastreduction,
             :inbounds_fastmath_simd_fastreduction,
             #= FIXME: broken :vmapreduce, =#
             #= FIXME: broken :turbo, =#)
    if what === :limits
        smin0, smax0 = linesearch_limits(Val(:ref),x,+,d,lo,hi)
        for impl in impls
            impl === :ref && continue
            smin, smax = linesearch_limits(Val(impl),x,+,d,lo,hi)
            (smin ≈ smin0 && smax ≈ smax0) || @warn "forward $impl -> ($smin,$smax), not ($smin0,$smax0)"
        end
        if bench
            for impl in impls
                print("  - $(rpad(string(impl), 45)) ->")
                @btime linesearch_limits($(Val(impl)),$x,+,$d,$lo,$hi);
            end
        end
        print("- backward direction:\n")
        smin0, smax0 = linesearch_limits(Val(:ref),x,-,d,lo,hi)
        for impl in impls
            impl === :ref && continue
            smin, smax = linesearch_limits(Val(impl),x,-,d,lo,hi)
            (smin ≈ smin0 && smax ≈ smax0) || @warn "forward $impl -> ($smin,$smax), not ($smin0,$smax0)"
        end
        if bench
            for impl in impls
                print("  - $(rpad(string(impl), 45)) ->")
                @btime linesearch_limits($(Val(impl)),$x,-,$d,$lo,$hi);
            end
        end
    elseif what === :stepmax
        smax0 = linesearch_stepmax(Val(:ref),x,+,d,lo,hi)
        for impl in impls
            (impl === :ref || impl === :dot) && continue
            smax = linesearch_stepmax(Val(impl),x,+,d,lo,hi)
            smax ≈ smax0 || @warn "forward $impl -> $smax, not $smax0"
        end
        if bench
            for impl in impls
                impl === :dot && continue
                print("  - $(rpad(string(impl), 45)) ->")
                @btime linesearch_stepmax($(Val(impl)),$x,+,$d,$lo,$hi);
            end
        end
        print("- backward direction:\n")
        smax0 = linesearch_stepmax(Val(:ref),x,-,d,lo,hi)
        for impl in impls
            (impl === :ref || impl === :dot) && continue
            smax = linesearch_stepmax(Val(impl),x,-,d,lo,hi)
            smax ≈ smax0 || @warn "forward $impl -> $smax, not $smax0"
        end
        if bench
            for impl in impls
                impl === :dot && continue
                print("  - $(rpad(string(impl), 45)) ->")
                @btime linesearch_stepmax($(Val(impl)),$x,-,$d,$lo,$hi);
            end
        end
    elseif what === :stepmin
        smin0 = linesearch_stepmin(Val(:ref),x,+,d,lo,hi)
        for impl in impls
            (impl === :ref || impl === :dot) && continue
            smin = linesearch_stepmin(Val(impl),x,+,d,lo,hi)
            smin ≈ smin0 || @warn "forward $impl -> $smin, not $smin0"
        end
        if bench
            for impl in impls
                impl === :dot && continue
                print("  - $(rpad(string(impl), 45)) ->")
                @btime linesearch_stepmin($(Val(impl)),$x,+,$d,$lo,$hi);
            end
        end
        print("- backward direction:\n")
        smin0 = linesearch_stepmin(Val(:ref),x,-,d,lo,hi)
        for impl in impls
            (impl === :ref || impl === :dot) && continue
            smin = linesearch_stepmin(Val(impl),x,-,d,lo,hi)
            smin ≈ smin0 || @warn "forward $impl -> $smin, not $smin0"
        end
        if bench
            for impl in impls
                impl === :dot && continue
                print("  - $(rpad(string(impl), 45)) ->")
                @btime linesearch_stepmin($(Val(impl)),$x,-,$d,$lo,$hi);
            end
        end
    end
end

end # module
