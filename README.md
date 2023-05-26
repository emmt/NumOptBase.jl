# Basic operations on variables for numerical optimization in Julia

[![Build Status](https://github.com/emmt/NumOptBase.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/NumOptBase.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/NumOptBase.jl?svg=true)](https://ci.appveyor.com/project/emmt/NumOptBase-jl)
[![Coverage](https://codecov.io/gh/emmt/NumOptBase.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/NumOptBase.jl)

`NumOptBase` implements efficient basic operations on variables for
multi-variate numerical optimization methods. It is similar to the `BLAS`
library for linear algebra methods.


## Operations on variables

The methods of `NumOptBase` are considered as low level methods and are not
automatically exported when `using NumOptBase`. This is also to avoid name
collision with other packages like `LinearAlgebra`.

The methods of `NumOptBase` may be extended by other packages to apply
numerical optimization methods to their own variables.


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


### Other operations

It is assumed that a few standard Julia methods are implemented in an efficient
way for the type of array storing the variables:

- `similar(x) -> y` to create a new array of variables `y` like `x`;
- `copyto!(dst, src) -> dst` to copy source variables `src` into destination
  variables `dst`;
- `fill!(x, α) -> x` to set all variables in `x` to the value `α`.
