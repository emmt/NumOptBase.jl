# Basic operations on variables for numerical optimization in Julia

[![Build Status](https://github.com/emmt/NumOptBase.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/NumOptBase.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/NumOptBase.jl?svg=true)](https://ci.appveyor.com/project/emmt/NumOptBase-jl)
[![Coverage](https://codecov.io/gh/emmt/NumOptBase.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/NumOptBase.jl)

`NumOptBase` implements efficient basic operations on variables for
multi-variate numerical optimization methods in [Julia](https://julialang.org).
It is similar to the `BLAS` library for linear algebra methods.

By leveraging the methods provided by `NumOptBase`, numerical optimization
methods can be written in a general way that is agnostic to the specific type
of arrays used to store the variables of the problem. Package
[`ConjugateGradient`](https://github.com/emmt/ConjugateGradient.jl) is such an
example. The methods of `NumOptBase` are thus intended to be extended by other
packages to apply numerical optimization methods to their own variables (that
is their own array types). For instance, dealing with
[`GPUArrays`](https://github.com/JuliaGPU/GPUArrays.jl) in `NumOptBase` is
currently under development.


## Variables in optimization methods

An optimization problem typically writes:

    minₓ f(x) s.t. x ∈ Ω

where `f: Ω → ℝ` is the objective function, `x` are the variables, and `Ω ⊆ ℝⁿ`
is the set of acceptable solutions with `n` the dimension of the problem.

It is assumed by this package that the variables `x` are stored in Julia
arrays. Depending on the problem, these arrays may be multidimensional but are
treated as real-valued *vectors* by the numerical optimization methods. For
efficiency, all entries of the arrays storing variables must have the same
floating-point type.

For now, quantities with units (such as those provided by the
[`Unitful`](https://github.com/PainterQubits/Unitful.jl) package) are not
supported. If your variables have units, you may consider using `reinterpret`
to remove units before calling the numerical optimization routines.


## Operations on variables

Some Julia methods are already available to deal with the variables of a
numerical optimization problem. For example, `similar` which may be called to
create a new array of variables provided the element type is some
floating-point real.

The `NumOptBase` package provides additional methods (described in what
follows) to operate on *variables* and which either require no additional
significant storage or store their result in an output array provided by the
caller. In that way, the storage requirements can be strictly controlled. All
these *public* methods are exported except `NumOptBase.copy!` which exists in
Julia base but with a different semantic.


### Norms

`norm1(x)`, `norm2(x)`, and `norminf(x)` respectively yield the ℓ₁, Euclidean,
and infinite norm of the variables `x`. They are similar to the norms in
`LinearAlgebra` except that they treat `x` as if it is has been flattened into
a vector of reals.


### Inner product

`inner(x, y)` yields the inner product (also known as *scalar product*) of the
variables `x` and `y` computed as expected by numerical optimization methods;
that is as if `x` and `y` are real-valued vectors and treating complex values
as pairs of reals in that respect. In other words, if `x` and `y` are
real-valued arrays, their inner product is given by:

    Σᵢ xᵢ⋅yᵢ

otherwise, if `x` and `y` are both complex-valued arrays, their inner product
is given by:

    Σᵢ (real(xᵢ)⋅real(yᵢ) + imag(xᵢ)⋅imag(yᵢ))

In the above pseudo-codes, index `i` runs over all indices of `x` and `y` which
may be multi-dimensional arrays but must have the same indices.

`inner(w, x, y)` yields:

    Σᵢ wᵢ⋅xᵢ⋅yᵢ

that is the *triple inner product* of the variables `w`, `x`, and `y` which
must all be real-valued arrays.


### Scaling, updating, and combining variables

`scale!(dst, α, x)` overwrites `dst` with `α⋅x` and returns `dst`. `α` is a
scalar while `dst` and `x` are arrays of the same size. If `iszero(α)` holds,
`dst` is zero-filled whatever the values in `x`.

`update!(x, β, y)` overwrites `x` with `x + β⋅y` and returns `x`. `β` is a
scalar while `x` and `y` are arrays of the same size. This is a shortcut for
`combine!(x,1,x,β,y)`.

`multiply!(dst, x, y)` overwrites `dst` with the element-wise multiplication
(also known as *Hadamard product*) of `x` by `y` and returns `dst`. `dst`, `x`,
and `y` must be arrays of the same size.

`combine!(dst, α, x, β, y)` overwrites `dst` with `α⋅x + β⋅y` and returns
`dst`. `α` and `β` must be real scalars while `dst`, `x`, and `y` must be
arrays of the same size.


### Applying mappings

The method:

``` julia
apply!(dst, f, args...) -> dst
```

overwrites the destination variables `dst` with the result of applying the
mapping `f` to arguments `args...`.

As of now, `apply!` only handles a few types of mappings:

- If `f` is an array, a generalized matrix-vector multiplication is applied to
  `args...` which must be a single array of variables.

- If `f` is `NumOptBase.Identity()`, the identity mapping is applied, that is
  the values of `src` are copied into `dst` (unless they are the same object).
  The constant `NumOptBase.Id` is the singleton object of type
  `NumOptBase.Identity`.

- If `f` is an instance of `NumOptBase.Diag`, an element-wise multiplication by
  `diag(f)` is applied.

The `NumOptBase.apply!` method shall be specialized in other argument types to
handle other cases.


### Other operations

`NumOptBase.copy!(dst, src)` overwrites the destination array `dst` with the
contents of the source array `src` throwing an error if they do not have the
same axes. If checking that the arguments have the same axes is not necessary,
the end-user may use `copyto!(dst, src)` or `copy!(dst, src)` which are basic
Julia methods.

It is assumed that a few standard Julia methods are implemented in an efficient
way for the type of array storing the variables:

- `similar(x) -> y` to create a new array of variables `y` like `x`;
- `copyto!(dst, src) -> dst` to copy source variables `src` into destination
  variables `dst`;
- `fill!(x, α) -> x` to set all variables in `x` to the value `α`.


## Bound constraints

`NumOptBase` provides some support for separable bound constraints on the
variables. With such constraints, the feasible set is defined by:

```
Ω = {x ∈ ℝⁿ | ℓ ≤ x ≤ u }
```

with `ℓ` and `u` the lower and upper bounds and where the comparisons (`≤`) is
taken element-wise. To represent the feasible set for bound constrained
`N`-dimensional variables of element type `T` in Julia is done by:

``` julia
Ω = BoundedSet{T,N}(ℓ, u)
```

where the lower and upper bounds, `ℓ` and `u`, may be specified as:
- `nothing` if the bound is unlimited;
- a scalar if the bound is the same for all variables;
- an array with the same axes as the variables.

For simplicity and type-stability, there are a number of restrictions which may
be alleviated in a high level interface:
- To avoid the complexity of managing all possibilities in the methods
  implementing bound constraints, bounds specified as arrays *conformable* with
  the variables are not directly supported. The caller may extend the array of
  bound values to the same size as the variables.
- Only `nothing` and the scalars `-∞` (for a lower bound) or `+∞` (for an upper
  bound) are considered as unlimited bounds even though all values of a lower
  (resp. upper) bound specified as an array may be `-∞` (resp. `+∞`).


### Projection on the feasible set

For any `x ∈ ℝⁿ`, the *projected variables* `xp ∈ Ω` are defined by:

```
xp = P(x) = argmin ‖y - x‖²   s.t.   y ∈ Ω
```

where `P` is the projection onto the feasible set `Ω`. In other words, `xp` is
the element of `Ω` that is the closest (in the least Euclidean distance sense)
to `x`.

The projected variables are computed by:

``` julia
project_variables!(xp, x, Ω)
```

which overwrites the destination `xp` with the projection of `x ∈ ℝⁿ` onto the
feasible set `Ω ⊆ ℝⁿ`.


### Projected direction and line-search

A number of numerical optimization methods proceed by iterations where, at the
`k`-th iteration, the next iterate writes:

```
xₖ₊₁ = P(xₖ ± αₖ⋅dₖ)   with   αₖ ≈ argmin f(P(xₖ ± α⋅dₖ))   s.t.   α ≥ 0
```

with `d ∈ ℝⁿ` a well chosen search direction and where, depending on the
numerical implementation, `±` is either `+` or `-` depending on whether the
variables vary as:

``` julia
x = P(xₖ + α⋅dₖ)
```

or:

``` julia
x = P(xₖ - α⋅dₖ)
```

along the path `α ≥ 0`. The `NumOptBase` package provides some methods to help
implementing such line-search methods.

For any feasible `x ∈ Ω` and search direction `d ∈ ℝⁿ`, the *projected
direction* `dp ∈ ℝⁿ` is defined by:

```
∀ α ∈ [0,ε], P(x ± α⋅d) = x ± α⋅dp
```

for some `ε > 0` and where `P` is the projection onto the feasible set `Ω`
previously defined. In other words, `dp` is the effective search direction in
`Ω` for any sufficiently small step size `α`.

The projected direction is computed by:

``` julia
project_direction!(dp, x, ±, d, Ω)
```

which overwrites the destination `dp` and where `±` is either `+` or `-`.

A closely related function is:

``` julia
unblocked_variables!(b, x, ±, d, Ω)
```

which overwrites the destination `b` with ones where variables in `x ∈ Ω` are
not blocked by the constraints implemented by `Ω` along direction `±d` and
zeros elsewhere. The projected direction `dp` and `b` are related by
`dp = b.*d`.

When line-searching, two specific values of the step length `α ≥ 0` are of
interest:

- `αₘᵢₙ ≥ 0` is the greatest nonnegative step length such that:

  ```
  α ≤ αₘᵢₙ  ⟹  P(x ± α⋅d) = x ± α⋅d
  ```

- `αₘₐₓ ≥ 0` is the least nonnegative step length such that:

  ```
  α ≥ αₘₐₓ  ⟹  P(x ± α⋅d) = P(x ± αₘₐₓ⋅d)
  ```

In other words, no bounds are overcome if `0 ≤ α ≤ αₘᵢₙ` and the projected
variables are all the same for any `α` such that `α ≥ αₘₐₓ`. The values of
`αₘᵢₙ` and/or `αₘₐₓ` can be computed by one of:

``` julia
αₘᵢₙ = linesearch_min_step(x, ±, d, Ω)
αₘₐₓ = linesearch_max_step(x, ±, d, Ω)
αₘᵢₙ, αₘₐₓ = linesearch_limits(x, ±, d, Ω)
```

Note that, for efficiency, `project_direction!`, `unblocked_variables!`,
`linesearch_min_step`, `linesearch_max_step`, and `linesearch_limits` assume
without checking that the input variables `x` are feasible, that is that `x ∈
Ω` holds.



## Extension to other array types

To extend the `NumOptBase` to other array types, some understanding of the
implementation of this package is needed. The **public methods** which can be
called by the end-users are summarized in the following table.

| Public method             | Description             | Remarks                            |
|:--------------------------|:------------------------|:-----------------------------------|
| `similar(x)`              | Yield an array like `x` |                                    |
| `zerofill!(dst)`          | Zero-fill `dst`         |                                    |
| `NumOptBase.copy!(dst,x)` | Copy `x` into `dst`     | See `copyto!` and `copy!` in Julia |
| `scale!(dst,α,x)`         | `dst = α*x`             |                                    |
| `update!(dst,α,x)`        | `dst += α*x`            |                                    |
| `combine!(dst,α,x,β,y)`   | `dst = α*x + β*y`       |                                    |
| `inner(x,y)`              | Inner product           |                                    |
| `inner(w,x,y)`            | Triple inner product    |                                    |
| `norm1(x)`                | ℓ₁ norm                 |                                    |
| `norm2(x)`                | Euclidean norm          |                                    |
| `norminf(x)`              | Infinite norm           |                                    |

In the above table and hereinafter, `dst`, `w`, `x`, and `y` denote arrays
(considered as *vectors*), `α` and `β` denote scalar reals, and all operations
and function calls are assumed to be done element-wise.

These public methods check their arguments (for having the same axes and thus
the same indices) and call one of the specialized methods listed below
depending on the operation, on the type of the array arguments, and on the
specific values of the multipliers `α` and `β`.

| Operation         | Specialized method                    | Remarks                  |
|:------------------|:--------------------------------------|:-------------------------|
| `dst = 0`         | `zerofill!(dst)`                      |                          |
| `dst = x`         | `unsafe_copy!(dst,x)`                 |                          |
| `dst = f(x)`      | `unsafe_map!(f,dst,x)`                |                          |
| `dst = f(x,y)`    | `unsafe_map!(f,dst,x,y)`              |                          |
| `dst = α*x`       | `unsafe_map!(αx(α,x),dst,x)`          | `α` is not 0, nor 1      |
| `dst = α*x + y`   | `unsafe_map!(αxpy(α,x),dst,x,y)`      | `α` is not 0, nor 1      |
| `dst = x + β*y`   | `unsafe_map!(αxpy(β,y),dst,y,x)`      | `β` is not 0, nor 1      |
| `dst = α*x + β*y` | `unsafe_map!(αxpβy(α,x,β,y),dst,x,y)` | neither `α` nor `β` is 0 |
| `dst = -x`        | `unsafe_map!(-,dst,x)`                |                          |
| `dst = x + y`     | `unsafe_map!(+,dst,x,y)`              |                          |
| `dst = x - y`     | `unsafe_map!(-,dst,x,y)`              |                          |
| `dst = x * y`     | `unsafe_map!(*,dst,x,y)`              |                          |
| `inner(x,y)`      | `unsafe_inner(x,y)`                   |                          |
| `inner(w,x,y)`    | `unsafe_inner(w,x,y)`                 |                          |
| `norm1(x)`        | `norm1(x))`                           |                          |
| `norm2(x)`        | `norm2(x)`                            |                          |
| `norminf(x)`      | `norminf(x)`                          |                          |

The prefix `unsafe_` means that the axes of arguments have been checked to be
compatible. Any scalar argument (`α` and `β`) shall never be zero and shall
have been converted to the correct floating-point type (this conversion is
automatically done by the constructors `αx`, `αxpy`, `αxmy`, and `αxpβy`). The
code of the high-level methods shall be simple enough for these methods to be
in-lined. This may lead to some optimizations (when the multipliers have
specific values like 0 or ±1).

Remarks:

- `unsafe_copy!(dst, x)` shall not be called when `dst` and `x` are the same
  object and amounts to calling `copyto!(dst, x)` by default but may be
  extended.

- `zerofill!(dst)` amounts to calling `fill!(dst, zero(eltype(dst)))` by
  default but `memset(pointer(dst), 0, sizeof(dst))` may be used for dense
  arrays.

- When `α` (resp. `β`) is zero, it is assumed that expression `α*x` (resp.
  `β*y`) is everywhere zero whatever the values of `x` (resp. `y`).

- It can be seen that a great deal of cases are handled by `unsafe_map!`. To
  avoid some overheads with closures and to allow for specialization of the
  code, `αx`, `αxpy`, `αxmy`, and `αxpβy` build callable objects which have
  specific types and which implement simple operation involving multipliers:

  ```julia
  f1 = αx(α,x)
  f1(xᵢ) -> α*xᵢ
  f2 = αxpy(α,x)
  f2(xᵢ,yᵢ) -> α*xᵢ + yᵢ
  f3 = αxmy(α,x)
  f3(xᵢ,yᵢ) -> α*xᵢ - yᵢ
  f4 = αxpβyy(α,x,β,y)
  f4(xᵢ,yᵢ) -> α*xᵢ + β*yᵢ
  ```

  where `xᵢ` and `yᵢ` denote an entry of `x` and `y`. These constructors take
  care of converting the multipliers `α` and `β`to the correct floating-point
  type. This the reason to provide arrays `x` and `y` along with their
  respective multiplier to the constructors.

To support specific array types or to optimize the operations for given array
types, it is sufficient to extend the specialized methods (the ones prefixed by
`unsafe_`) and the methods that compute the norms. Specializing the method
`zerofill!` is not mandatory as the default version shall work for all array
types.

You may have a look in the files
[ext/NumOptBaseLoopVectorizationExt.jl](ext/NumOptBaseLoopVectorizationExt.jl)
and [ext/NumOptBaseCUDAExt.jl](ext/NumOptBaseCUDAExt.jl) which respectively
extend `NumOptBase` to use AVX loop vectorization and Cuda GPU arrays.
