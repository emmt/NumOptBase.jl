# Efficient vectorized operations

## Units


## Vectorized operations

In conjugate gradient method:

- `similar(x)` create workspace array;
- `vdot(x,y)` inner product of `x` and `y`;
- `@vops x += alpha*p` update variables;
- `@vops r -= alpha*q` update residuals;
- `apply!(q, A, p)` apply "matrix" `A`;
- `@vops p = z` copy variables;
- `@vops p = z + beta*p` next search direction;

In *Spectral Projected Gradient Method* (SPG):

- `similar(x)` create workspace array;
- `fill!(x, 0)` zero-fill variables;
- `vdot(x,y)` inner product of `x` and `y`;
- `fg!(x, g)` compute objective function and gradient;
- `prj!(dst, src)` project variables to feasible set;
- `@vops pg = (x - prj!(pg, x - eta*g))/eta` compute *projected gradient* using
  `pg` to temporary store the projection of `x - pg/eta`;
- `vnorm2(pg)` Euclidean norm;
- `vnorminf(pg)` infinite norm;
- `@vops s = x - x0` effective step;
- `@vops y = g - g0` gradient change;
- `@vops x0 = x` copy variables;

In VMLMB method:

- Hadamard product (element-wise multiplication);


Standard methods:

- `similar(arr) -> newarr` to create a new array `newarr` like `arr`;
- `copyto!(dst, src) -> dst` to copy array `src` into `dst`;
- `fill!(arr, val) -> arr` to fill array `arr` with value `val`;

```julia
```

With `map!`, using a callable object works fine for regular arrays but not for
GPU arrays.

```julia
struct Combine{F,T} <: Function
    func::F
    alpha::T
    beta::T
end
@inline (obj::Combine)(x, y) = obj.func(obj.alpha, x, obj.beta, y)
@inline axpby_xpy(a, x, b, y) = x + y
@inline axpby_xmy(a, x, b, y) = x - y
@inline axpby_axpy(a, x, b, y) = a*x + y
@inline axpby_axmy(a, x, b, y) = a*x - y
@inline axpby_xpby(a, x, b, y) = x + b*y
@inline axpby_axpby(a, x, b, y) = a*x + b*y

function combine!(dst::AbstractArray{T,N},
                  α::Real, x::AbstractArray{T,N},
                  β::Real, y::AbstractArray{T,N}) where {T,N}
    @assert axes(dst) == axes(x) == axes(y)
    unsafe_combine!(dst, α, x, β, y)
    return dst
end

```
