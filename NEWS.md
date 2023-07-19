# User visible changes in `NumOptBase`

# Version 0.1.10

- `NumOptBase.scale!(α,x)` and `NumOptBase.scale!(x,α)` are shortcuts to
  `NumOptBase.scale!(x,α,x)`.

- Extend `convert` for ``NumOptBase.Diag`.

# Version 0.1.9

- Use `ccall` instead of `@ccall` for Julia < 1.5.

# Version 0.1.8

- Re-write splitting in low/high level methods.

- Accelerate `norm1` and `norminf` with `LoopVectorization`.

# Version 0.1.7

- Fix dependencies:
  - Julia ≥ 1.2 is needed for `mapreduce` with multiple arguments.
  - Julia ≥ 1.3 is needed for `CUDA`.

- Drop dependency on `Unitless`.

# Version 0.1.6

- Implement operations for GPU arrays provided by the
  [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) package.

- In `LoopVectorization` extension: fix norms and restrict arguments to real-valued
  arrays.

# Version 0.1.5

- Fix norms for complex-valued variables.

- Norms of scalars.

- Speed-up infinite norm when `LoopVectorization` is not loaded.

# Version 0.1.4

- Fix extend of `LoopVectorization` to more other basic operations.

# Version 0.1.3

- Extend  `LoopVectorization` to more other basic operations.

- Avoid using `LoopVectorization` if the `@turbo` macro is not defined (i.e.
  for Julia < 1.5 or `LoopVectorization` < 0.12.22).

# Version 0.1.2

- Add `LoopVectorization` to speed up some operations as a package extension
  (for Julia ≥ 1.9) or using
  [`Requires`](https://github.com/JuliaPackaging/Requires.jl) (for Julia <
  1.9).

# Version 0.1.1

- Add a few cases that can be handled by `apply!`: the identity
  `NumOptBase.Id`, diagonal linear mappings built by `NumOptBase.Diag`, and
  generalized matrix-vector multiplication.
