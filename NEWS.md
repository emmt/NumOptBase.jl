# User visible changes in `NumOptBase`

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
