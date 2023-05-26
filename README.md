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
is the set of acceptable solution with `n` the dimension of the problem.

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

The methods of `NumOptBase` are considered as low level methods and are not
automatically exported when `using NumOptBase`. This is also to avoid name
collision with other packages like `LinearAlgebra`.

Except `similar` which may be called to create a new array of variables, all
other methods either require no additional significant storage or store their
result in an output array provided by the caller. In that way, the storage
requirements can be strictly controlled.


### Norms

`NumOptBase.norm1(x)`, `NumOptBase.norm2(x)`, and `NumOptBase.norminf(x)`
respectively yield the ℓ₁, Euclidean, and infinite norm of the variables `x`.
They are similar to the norms in `LinearAlgebra` except that they treat `x` as
if it is has been flattened into a vector of reals.


### Inner product

`NumOptBase.inner(x, y)` yields the inner product (also known as *scalar
product*) of the variables `x` and `y` computed as expected by numerical
optimization methods; that is as if `x` and `y` are real-valued vectors and
treating complex values as pairs of reals in that respect. In other words, if
`x` and `y` are real-valued arrays, their inner product is given by:

    Σᵢ xᵢ⋅yᵢ

otherwise, if `x` and `y` are both complex-valued arrays, their inner product
is given by:

    Σᵢ (real(xᵢ)⋅real(yᵢ) + imag(xᵢ)⋅imag(yᵢ))

In the above pseudo-codes, index `i` runs over all indices of `x` and `y` which
may be multi-dimensional arrays but must have the same indices.

`NumOptBase.inner(w, x, y)` yields

    Σᵢ wᵢ⋅xᵢ⋅yᵢ

that is the *triple inner product* of the variables `w`, `x`, and `y` which
must all be real-valued arrays.


### Scaling, updating, and combining variables

`NumOptBase.scale!(dst, α, x)` overwrites `dst` with `α⋅x` and returns `dst`.
`α` is a scalar while `dst` and `x` are variables of the same size. If
`iszero(α)` holds, `dst` is zero-filled whatever the values in `x`.

`NumOptBase.update!(x, β, y)` overwrites `x` with `x + β⋅y` and returns `x`.
`β` is a scalar while `x` and `y` are variables of the same size. This is a
shortcut for `NumOptBase.combine!(x,1,x,β,y)`.

`NumOptBase.multiply!(dst, x, y)` overwrites `dst` with the element-wise
multiplication (also known as *Hadamard product*) of `x` by `y` and returns
`dst`. `dst`, `x`, and `y` must be variables of the same size.

`NumOptBase.combine!(dst, α, x, β, y)` overwrites `dst` with `α⋅x + β⋅y` and
returns `dst`. `α` and `β` must be real scalars while `dst`, `x`, and `y` must
be variables of the same size.


### Applying mappings

The method:

``` julia
NumOptBase.apply!(dst, f, args...) -> dst
```

overwrites the destination `dst` variables with the result of applying the
mapping `f` to arguments `args...`.

As of now, `NumOptBase.apply!` only handles a few types of mappings:

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

It is assumed that a few standard Julia methods are implemented in an efficient
way for the type of array storing the variables:

- `similar(x) -> y` to create a new array of variables `y` like `x`;
- `copyto!(dst, src) -> dst` to copy source variables `src` into destination
  variables `dst`;
- `fill!(x, α) -> x` to set all variables in `x` to the value `α`.
