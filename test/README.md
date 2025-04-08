# Implementing fast vectorized operations in Julia

Vectorized operations exploit [SIMD (single instruction, multiple
data)](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) to accelerate
computations onto array elements. This kind of optimization is triggered by the macros
`@inbounds` and `@inbounds @simd` of Julia, possibly associated with `@fastmath`, and by
the macro `@turbo` provided by
[LoopVectorization](https://github.com/JuliaSIMD/LoopVectorization.jl).

## `@fastmath`

Julia `@fastmath` macro (in front of an expression) triggers optimizations and makes
assumptions that may be beneficial to accelerate computations. There is however a price to
pay: with `@fastmath`, some operations no longer comply with IEEE rules for floating
point arithmetic. These rules are mostly suitable but may yield slow code.

Based on Julia code for `@fastmath` (in
[`base/fastmath.jl`](https://github.com/JuliaLang/julia/blob/master/base/fastmath.jl)) and
on the tests implemented in [`test/fastmath.jl`](fastmath.jl), the following points
summarize what happens with `@fastmath`:

- Evaluations of expressions may be re-ordered or simplified according to the properties
  of exact arithmetic operations (like associativity) but which not generally hold for
  their numerical counterparts. This may have disastrous effects on the precision or on
  some algorithms.

- It is assumed that no value can be a NaN, subnormal, or infinite and that there are no
  signed zeros. Hence, `isnan(x)`, `issubnormal(x)`, and `isinf(x)` always yield `false`
  with `@fastmath`. As a consequence, `isfinite(x)` always yield `true` with `@fastmath`.

- Comparison involving a NaN may be wrong. For example, with `@fastmath`, `==`, `<`, `<=`,
  `>`, and `>=` all yield `true` if any of the operands is a NaN. However, `===`,
  `isequal`, and `isless` yield a correct result. As another example, in `cmp(x,y)`,
  without `@fastmath`, NaNs are considered as the largest possible values, while, with
  `@fastmath`, NaNs are considered as equal to any value.

- Methods `iszero(x)`, `isone(x)`, `isless(x,y)`, and `isequal(x,y)` yield the same result
  whether `@fastmath` is applied or not.

- Math functions may yield a result that is slightly less precise with `@fastmath` than
  without and may return `NaN` instead of throwing a `DomainError` exception. This may be
  more convenient or not. Examples:

  ```julia
  julia> @fastmath(exp(-1.0)) - exp(-1.0)
  -5.551115123125783e-17

  julia> try sqrt(-1.0); catch ex; ex; end
  DomainError(-1.0, "sqrt was called with a negative real argument but will only return a complex result if called with a complex argument. Try sqrt(Complex(x)).")

  julia> try @fastmath(sqrt(-1.0)); catch ex; ex; end
  NaN
  ```

- Some math functions, like `hypot`, yield a more appropriate result with `@fastmath` than
  without. For example:

  ```julia
  julia> hypot(Inf,NaN)
  Inf

  julia> @fastmath(hypot(Inf,NaN))
  NaN
  ```

Observed behaviors with `@fastmath`:

- **Ordering comparison operators:** `>`, `>=`, `<`, and `<=` all yield `true` if at least
  one of the operands is `NaN`. However, `isless` is correct.

- **Equality:** `==` yields `true` and `!=` yields `false` if at least one of the operands
  is `NaN`. However, `===`, `isequal`, `isone`, and `iszero` are correct.

- **Predicates:** `isnan(x)`, `issubnormal(x)`, and `isinf(x)` yield `false` whatever `x`.
    However, `isequal(x,y)`, `isone(x)`, and `iszero(x)` are correct.

- **Min/Max** `min(x,y)` and `max(x,y)` yield a result that depend on the order of the
  arguments if at least one is `NaN`. I presume that this affects `clamp(x,lo,hi)` and
  `minmax(x,y)`.

- **Infinite** As far as I can tell, `±∞` seems to behave the same with and without
  `@fastmath`.

These imply that you want to make sure that your code **does not produces NaNs**. There
are however some good news:

- `NaN` and `Inf` are correctly produced by arithmetic operations. For example:

  ```julia
  julia> @fastmath(0.0/0.0)
  NaN

  julia> @fastmath(1.0/0.0)
  Inf

  julia> @fastmath(-1.0/0.0)
  -Inf

  julia> @fastmath(Inf/0.0)
  Inf

  julia> @fastmath(Inf/Inf)
  NaN

  julia> @fastmath(Inf + Inf)
  Inf

  julia> @fastmath(Inf - Inf)
  NaN
  ```


## Links

- The Julia code for `@fastmath` is in
  [`base/fastmath.jl`](https://github.com/JuliaLang/julia/blob/master/base/fastmath.jl).

- [Simon's notes]( https://simonbyrne.github.io/notes/fastmath/) also discusses the use of
  `@fastmath` in Julia.

- Noting that Julia is based on the LLVM like CLang, the `@fastmath` macro is closely
  related to the `-ffast-math` option of CLang. Although GCC and CLang are different
  compilers, some useful information for GCC can be found on [this Stackoverflow
  discussion](https://stackoverflow.com/questions/7420665/what-does-gccs-ffast-math-actually-do)
