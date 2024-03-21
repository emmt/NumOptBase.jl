# User visible changes in `NumOptBase`

## Version 0.2.3

- `isempty(Ω)` yields whether the bounded set `Ω` is empty, that is infeasible.

- `x ∈ Ω` yields whether variables `x` belongs to the bounded set `Ω`.

- Fallback `zerofill!(A)` is simply `fill!(A, zero(eltype(A)))`.

- `update!(dst,α,x,y)` uses `engine(dst,x,y)` to determine engine.

## Version 0.2.2

- In `NumOptBase.convert_multiplier(α, A)`, the scalar `α` may be any `Number`
  (not just a `Real`).

- The type `T` of the value returned by `norm1`, `norm2`, `norminf`, and
  `inner` may be specified as a leading argument of these functions.

- `combine!(dst, x, ±, y)` with `±` being either `+` or `-`
   can be used to do `dst[i] = x[i] ± y[i]` for all indices `i`.

- `LinearAlgebra.lmul!` extended so that `lmul!(dst, Diag(w), x)` and
   `lmul!(dst, Id, x)` work as expected (i.e. respectively `multiply!(dst, w, x)`
   and `NumOptBase.copy!(dst, x)`).

## Version 0.2.1

- Extend `update!` so that `update!(x, β, y, z)` returns `x` overwritten with
  `x + β⋅y⋅z` performed element-wise. Arguments `x`, `y`, and `z` are arrays of
  the same size while `β` is a scalar.

- Implement conversions between bounded sets.

- New `Projector` constructor to build a projector from a feasible set.

## Version 0.2.0

This branch adds bound constraints.

- New types for bound constraints: `Bound{T,N}` and `BoundedSet{T,N,L,U}` with
  `T` and `N` the element type and number of dimensions of the variables, `L`
  and `U` the types of the lower and upper bounds.

- New methods for bound constraints: `project_variables!`,
  `project_direction!`, `unblocked_variables!`, `linesearch_limits`,
  `linesearch_stepmin`, and `linesearch_stepmax`.

## Version 0.1.15

- Fix imports in turbo code.
- Define alias `NumOptBase.SimdArray{T,N}` for array types suitable for `@simd`
  loop optimization. For now, this is an alias to `AbstractyArray{T,N}` as it
  is assumed that the `@simd` macro is smart enough to decide whether SIMD loop
  optimization can be used considering the types of the arrays involved in the
  loop.

## Version 0.1.14

- Export public methods (but `copy!` which has a different semantic than in
  Julia).
- Alias `NumOptBase.TurboArray{T,N}` for array types suitable for `@turbo` loop
  optimization.
- Fix tests for non-indexable arrays such as GPU arrays.
- Extend compatibility to `CUDA` package version 5.

## Version 0.1.13

- Simplify concept of *engines* by only using abstract types (this is all what
  is needed to convey a hierarchy).

## Version 0.1.12

- New concept of *engines* having different implementations of the method
  co-exist in the same Julia session and be either automatically selected or
  chosen by the user.

## Version 0.1.11

- New macro `NumOptBase.@vectorize` to compile loops with various
  optimizations.

## Version 0.1.10

- `NumOptBase.scale!(α,x)` and `NumOptBase.scale!(x,α)` are shortcuts to
  `NumOptBase.scale!(x,α,x)`.

- Extend `convert` for `NumOptBase.Diag`.

- Use `TypeUtils` instead of our own `NumOptBase.as` method.

## Version 0.1.9

- Use `ccall` instead of `@ccall` for Julia < 1.5.

## Version 0.1.8

- Re-write splitting in low/high level methods.

- Accelerate `norm1` and `norminf` with `LoopVectorization`.

## Version 0.1.7

- Fix dependencies:
  - Julia ≥ 1.2 is needed for `mapreduce` with multiple arguments.
  - Julia ≥ 1.3 is needed for `CUDA`.

- Drop dependency on `Unitless`.

## Version 0.1.6

- Implement operations for GPU arrays provided by the
  [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) package.

- In `LoopVectorization` extension: fix norms and restrict arguments to real-valued
  arrays.

## Version 0.1.5

- Fix norms for complex-valued variables.

- Norms of scalars.

- Speed-up infinite norm when `LoopVectorization` is not loaded.

## Version 0.1.4

- Fix extend of `LoopVectorization` to more other basic operations.

## Version 0.1.3

- Extend  `LoopVectorization` to more other basic operations.

- Avoid using `LoopVectorization` if the `@turbo` macro is not defined (i.e.
  for Julia < 1.5 or `LoopVectorization` < 0.12.22).

## Version 0.1.2

- Add `LoopVectorization` to speed up some operations as a package extension
  (for Julia ≥ 1.9) or using
  [`Requires`](https://github.com/JuliaPackaging/Requires.jl) (for Julia <
  1.9).

## Version 0.1.1

- Add a few cases that can be handled by `apply!`: the identity
  `NumOptBase.Id`, diagonal linear mappings built by `NumOptBase.Diag`, and
  generalized matrix-vector multiplication.
