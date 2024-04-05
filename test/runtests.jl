using Test
using LinearAlgebra
using StructuredArrays
using StructuredArrays: value
using TypeUtils
using NumOptBase
using NumOptBase:
    check_axes,
    engine,
    is_bounding,
    is_bounding_above,
    is_bounding_below,
    only_arrays,
    step_from_bounds,
    step_from_lower_bound,
    step_from_upper_bound,
    step_to_bounds,
    step_to_lower_bound,
    step_to_upper_bound,
    stepmax_reduce,
    stepmin_reduce

# Automatically use LoopVectorization if Julia is sufficiently recent. NOTE: It
# is always possible to manually load LoopVectorization before testing).
VERSION ≥ v"1.5" && using LoopVectorization

isdefined(@__MODULE__,:Generic) || include("Generic.jl")

@testset "NumOptBase" begin
    array_types = Array{Type{<:AbstractArray}}(undef, 0)
    push!(array_types, Array, Generic.OtherArray)

    floats = (Float64, Float32)

    @testset "Floating-point behavior ($T)" for T in floats
        @test  Generic.same_float(+zero(T), zero(T))
        @test !Generic.same_float(-zero(T), zero(T))
        @test  Generic.same_float(T(NaN), T(NaN))
        @test  Generic.same_float((+zero(T))/zero(T), T(NaN))
        @test  Generic.same_float((-zero(T))/zero(T), T(NaN))
        @test  Generic.same_float(T(+Inf), typemax(T))
        @test  Generic.same_float(T(-Inf), typemin(T))
        @test  Generic.same_float(one(T)/(+zero(T)), T(+Inf))
        @test  Generic.same_float(one(T)/(-zero(T)), T(-Inf))
        @test  Generic.same_float(zero(T), T(0))
        @test  Generic.same_float(one(T), T(1))

        # The 7 different possible floating-point values (-1 and +1
        # representing any finite and respectively negative or positive
        # value).
        v = T[-Inf, -1.0, -0.0, +0.0, +1.0, +Inf, NaN]
        # Ratio of floating-point values.
        r = v' ./ v
        # Expected result for the ratio of floating-point values.
        s = T[#=(1,1:7)=#  NaN   0.0   0.0  -0.0  -0.0   NaN   NaN;
              #=(2,1:7)=#  Inf   1.0   0.0  -0.0  -1.0  -Inf   NaN;
              #=(3,1:7)=#  Inf   Inf   NaN   NaN  -Inf  -Inf   NaN;
              #=(4,1:7)=# -Inf  -Inf   NaN   NaN   Inf   Inf   NaN;
              #=(5,1:7)=# -Inf  -1.0  -0.0   0.0   1.0   Inf   NaN;
              #=(6,1:7)=#  NaN  -0.0  -0.0   0.0   0.0   NaN   NaN;
              #=(7,1:7)=#  NaN   NaN   NaN   NaN   NaN   NaN   NaN]
        @test eltype(r) === eltype(s) === T
        @test Generic.same_float(r, s; verb=true)
        if isdefined(Main, :CUDA)
            vp = Main.CUDA.CuArray(v)
            r = Array(vp' ./ vp)
            @test eltype(r) === eltype(s) === T
            @test Generic.same_float(r, s; verb=true)
        end
    end

    @testset "Type Utilities" begin
        as = NumOptBase.as
        @test as(Int, 3) === 3
        @test as(typeof(sin), sin) === sin
        @test as(Float32, 3) === 3.0f0
        @test as(Float32, π) === Float32(π)

        floating_point_type = NumOptBase.floating_point_type
        @testset "floating_point_type($T)" for (T,F) in ((Int, float(Int)),
                                                         (Float16, Float16),
                                                         (Float32, Float32),
                                                         (Float64, Float64),
                                                         (BigFloat, BigFloat))
            @test floating_point_type(T) === F
            @test floating_point_type(zero(T)) === F
            @test floating_point_type(Complex{T}) === F
            @test floating_point_type(zero(Complex{T})) === F
        end
        @test_throws Exception floating_point_type("hello")
    end

    @testset "Array Utilities ($A)" for A in array_types
        w = convert(A, collect(-1:4))
        x = convert(A, zeros(Float64, 2,3))
        y = convert(A, ones(Float32, size(x)))
        E = @inferred engine(w,x,y)
        if A <: Array
            @test E === NumOptBase.TurboLoopEngine
        elseif A <: Generic.OtherArray
            @test E === NumOptBase.SimdLoopEngine
        end
        @test () === @inferred only_arrays()
        @test () === @inferred only_arrays(π)
        @test (x,) === @inferred only_arrays(x)
        @test (x,) === @inferred only_arrays(sin, x, 1)
        @test (x, y) === @inferred only_arrays(x, y)
        @test (x, y) === @inferred only_arrays(x, 1, y, nothing)
        @test (x, y) === @inferred only_arrays(sin, x, 1, y)
        @test (w, x, y) === @inferred only_arrays(w, x, y)
        @test (w, x, y) === @inferred only_arrays(w, x, 1, y)
        @test (w, x, y) === @inferred only_arrays(0, w, x, 1, y)
        #=
        @test nothing === @inferred check_axes()
        @test nothing === @inferred check_axes(x)
        @test nothing === @inferred check_axes(x, y)
        @test_throws DimensionMismatch check_axes(w, x)
        @test_throws DimensionMismatch check_axes(x, x')
        =#
    end

    eltypes = (Float64, Complex{Float32})
    dims = (3,4,5)
    alphas = (0, 1, -1, 2.1)
    betas = (0, 1, -1, -1.7)
    @testset "Array Operations ($A{$T,$(length(dims))})" for A in array_types, T ∈ eltypes
        w_ref = rand(T, dims)
        x_ref = rand(T, dims)
        y_ref = rand(T, dims)
        tmp = Array{T}(undef, dims)
        w = convert(A, w_ref)
        x = convert(A, x_ref)
        y = convert(A, y_ref)
        z = similar(x)
        E = NumOptBase.LoopEngine # most basic engine
        F = (real(T) === Float64 ? Float32 : Float64) # other floating-point type
        rtol = 2eps(Float32)
        @testset "norm1" begin
            let xᵢ = first(x_ref), res = @inferred norm1(xᵢ)
                @test typeof(res) === real(T)
                @test res ≈ Generic.norm1(xᵢ)
            end
            let res_ref = Generic.norm1(x_ref)
                let res = @inferred norm1(x)
                    @test typeof(res) === real(T)
                    @test res ≈ res_ref
                end
                let res = @inferred norm1(E, x)
                    @test typeof(res) === real(T)
                    @test res ≈ res_ref
                end
                let res = @inferred norm1(F, x)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
                let res = @inferred norm1(F, E, x)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
                let res = @inferred norm1(E, F, x)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
            end
        end
        @testset "norm2" begin
            let xᵢ = first(x_ref), res = @inferred norm2(xᵢ)
                @test typeof(res) === real(T)
                @test res ≈ Generic.norm2(xᵢ)
            end
            let res_ref = Generic.norm2(x_ref)
                let res = @inferred norm2(x)
                    @test typeof(res) === real(T)
                    @test res ≈ res_ref
                end
                let res = @inferred norm2(E, x)
                    @test typeof(res) === real(T)
                    @test res ≈ res_ref
                end
                let res = @inferred norm2(F, x)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
                let res = @inferred norm2(F, E, x)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
                let res = @inferred norm2(E, F, x)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
            end
        end
        @testset "norminf" begin
            let xᵢ = first(x_ref), res = @inferred norminf(xᵢ)
                @test typeof(res) === real(T)
                @test res ≈ Generic.norminf(xᵢ)
            end
            let res_ref = Generic.norminf(x_ref)
                let res = @inferred norminf(x)
                    @test typeof(res) === real(T)
                    @test res ≈ res_ref
                end
                let res = @inferred norminf(E, x)
                    @test typeof(res) === real(T)
                    @test res ≈ res_ref
                end
                let res = @inferred norminf(F, x)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
                let res = @inferred norminf(F, E, x)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
                let res = @inferred norminf(E, F, x)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
            end
        end
        @testset "inner product" begin
            let res_ref = Generic.inner(x_ref, y_ref)
                let res = @inferred inner(x, y)
                    @test typeof(res) === real(T)
                    @test res ≈ res_ref
                end
                let res = @inferred inner(E, x, y)
                    @test typeof(res) === real(T)
                    @test res ≈ res_ref
                end
                let res = @inferred inner(F, x, y)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
                let res = @inferred inner(E, F, x, y)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
                let res = @inferred inner(F, E, x, y)
                    @test typeof(res) === F
                    @test res ≈ F(res_ref) rtol=rtol
                end
            end
            if T <: Complex
                @test_throws Exception inner(w, x, y)
            else
                let res_ref = Generic.inner(w_ref, x_ref, y_ref)
                    let res = @inferred inner(w, x, y)
                        @test typeof(res) === real(T)
                        @test res ≈ res_ref
                    end
                    let res = @inferred inner(E, w, x, y)
                        @test typeof(res) === real(T)
                        @test res ≈ res_ref
                    end
                    let res = @inferred inner(F, w, x, y)
                        @test typeof(res) === F
                        @test res ≈ F(res_ref) rtol=rtol
                    end
                    let res = @inferred inner(E, F, w, x, y)
                        @test typeof(res) === F
                        @test res ≈ F(res_ref) rtol=rtol
                    end
                    let res = @inferred inner(F, E, w, x, y)
                        @test typeof(res) === F
                        @test res ≈ F(res_ref) rtol=rtol
                    end
                end
            end
        end
        @testset "zero-fill" begin
            copyto!(z, x)
            let res = @inferred zerofill!(z)
                @test res === z
                @test all(iszero, z)
            end
        end
        @testset "copy" begin
            zerofill!(z)
            let res = @inferred NumOptBase.copy!(z, x)
                @test res === z
                @test z == x
            end
        end
        @testset "scale! α = $α" for α in alphas
            let res = @inferred scale!(z, α, x)
                @test res === z
                @test copyto!(tmp, res) ≈ (@. α*x_ref)
            end
            let res = @inferred scale!(α, copyto!(z, x))
                @test res === z
                @test copyto!(tmp, res) ≈ (@. α*x_ref)
            end
            let res = @inferred scale!(copyto!(z, x), α)
                @test res === z
                @test copyto!(tmp, res) ≈ (@. α*x_ref)
            end
        end
        @testset "multiply!" begin
            let res = @inferred multiply!(z, x, y)
                @test res === z
                @test copyto!(tmp, res) ≈ (@. x_ref*y_ref)
            end
        end
        @testset "update! α = $α" for α in alphas
            let res = @inferred update!(copyto!(z, y), α, x)
                @test res === z
                @test copyto!(tmp, res) ≈ (@. y_ref + α*x_ref)
            end
            let res = @inferred update!(copyto!(z, w), α, x, y)
                @test res === z
                @test copyto!(tmp, res) ≈ (@. w_ref + α*x_ref*y_ref)
            end
        end
        @testset "combine! α = $α, β = $β" for α in alphas, β ∈ betas
            let res = @inferred combine!(z, α, x, β, y)
                @test res === z
                @test copyto!(tmp, res) ≈ (@. α*x_ref + β*y_ref)
            end
        end
        @testset "combine! (±)" begin
            let res = @inferred combine!(z, x, +, y)
                @test res === z
                @test copyto!(tmp, res) ≈ (@. x_ref + y_ref)
            end
            let res = @inferred combine!(z, x, -, y)
                @test res === z
                @test copyto!(tmp, res) ≈ (@. x_ref - y_ref)
            end
        end
    end

    @testset "Array Operators ($A{$T,$(length(dims))})" for A in array_types, T ∈ floats
        Diag = NumOptBase.Diag
        Id = NumOptBase.Id
        dims = (2,3,4)
        N = length(dims)
        w = convert(A, rand(T, dims))
        x = convert(A, rand(T, dims))
        y = similar(x)
        z = similar(x)
        D = @inferred Diag(w)
        @test diag(D) === w
        @test apply!(z, D, x) ≈ w .* x
        @test convert(Diag, D) === D
        @test convert(typeof(D), D) === D
        @test convert(Diag{eltype(diag(D))}, D) === D
        @test convert(Diag{eltype(diag(D)),ndims(diag(D))}, D) === D
        @test convert(Diag{eltype(diag(D)),ndims(diag(D)),typeof(diag(D))}, D) === D
        @test_throws Exception convert(Diag{eltype(diag(D)),ndims(diag(D))+1}, D)
        let A = @inferred convert(Diag{Float64}, D)
            @test diag(A) ≈ diag(D)
            @test eltype(diag(A)) === Float64
         end
        let A = @inferred convert(Diag{Float64,N}, D)
            @test diag(A) ≈ diag(D)
            @test eltype(diag(A)) === Float64
        end
        let A = @inferred convert(Diag{Float64,N,Array{Float64,N}}, D)
            @test diag(A) ≈ diag(D)
            @test eltype(diag(A)) === Float64
        end
        @test lmul!(z, D, x) == apply!(z, D, x)
        @test apply!(z, Id, x) == x
        @test lmul!(z, Id, x) == apply!(z, Id, x)
    end

    # Utilities for bounds.
    @testset "is_bounding_below" begin
        @test !is_bounding_below(nothing)
        @test  is_bounding_below(1.3)
        @test  is_bounding_below(typemax(Float32))
        @test !is_bounding_below(typemin(Float64))
        @test  is_bounding_below(UniformArray(typemax(Float64), 2, 3, 4))
        @test !is_bounding_below(UniformArray(typemin(Float32), 2, 3, 4))
        @test  is_bounding_below(fill(typemin(Float64), 2, 3))
    end
    @testset "is_bounding_above" begin
        @test !is_bounding_above(nothing)
        @test  is_bounding_above(1.3)
        @test !is_bounding_above(typemax(Float32))
        @test  is_bounding_above(typemin(Float64))
        @test !is_bounding_above(UniformArray(typemax(Float64), 2, 3, 4))
        @test  is_bounding_above(UniformArray(typemin(Float32), 2, 3, 4))
        @test  is_bounding_above(fill(typemax(Float64), 2, 3))
    end
    @testset "stepmax_reduce" begin
        # Result shall not depend on order of arguments.
        @test stepmax_reduce(NaN, 2.0) === 2.0
        @test stepmax_reduce(1.0, NaN) === 1.0
        @test stepmax_reduce(NaN, NaN) === NaN
        @test stepmax_reduce(1.0, 2.0) === 2.0
        @test stepmax_reduce(2.0, 1.0) === 2.0
    end
    @testset "stepmin_reduce" begin
        # Result shall not depend on order of arguments.
        @test stepmin_reduce(NaN, 2.0) === 2.0
        @test stepmin_reduce(1.0, NaN) === 1.0
        @test stepmin_reduce(NaN, NaN) === NaN
        @test stepmin_reduce(1.0, 2.0) === 1.0
        @test stepmin_reduce(2.0, 1.0) === 1.0
    end
    @testset "step_to_bounds(±)" begin
        @test step_to_bounds(     +) === step_to_bounds
        @test step_to_bounds(     -) === step_from_bounds
        @test step_to_lower_bound(+) === step_to_lower_bound
        @test step_to_lower_bound(-) === step_from_lower_bound
        @test step_to_upper_bound(+) === step_to_upper_bound
        @test step_to_upper_bound(-) === step_from_upper_bound
    end

    # The values of x, d, l, and l are chosen to have exact results.
    x = 1.0
    @testset "step_to_bounds(x=$x, d=$d, l=$l, u=$u)" for d in (+2.0, 0.0, -2.0),
        l in (-5.0, -Inf, NaN), u in (9.0, +Inf, NaN)
        # NaN for a bound in the direction of search is interpreted in the
        # tests as unbounded, but the bound on the opposite side may be
        # anything (including NaN) because the result should not depend on it.
        if d > zero(d)
            if isnan(u) # test as if unbounded above
                r = +Inf
                @test step_to_lower_bound(  x, +,  d, l   ) === r
                @test step_to_lower_bound(  x,     d, l   ) === r
                @test step_to_lower_bound(  x, -, -d, l   ) === r
                @test step_from_lower_bound(x,    -d, l   ) === r
            else
                r = u < typemax(u) ? (u - x)/d : +Inf
                @test step_to_bounds(       x, +,  d, l, u) === r
                @test step_to_bounds(       x,     d, l, u) === r
                @test step_to_bounds(       x, -, -d, l, u) === r
                @test step_from_bounds(     x,    -d, l, u) === r
                @test step_to_upper_bound(  x, +,  d,    u) === r
                @test step_to_upper_bound(  x,     d,    u) === r
                @test step_to_upper_bound(  x, -, -d,    u) === r
                @test step_from_upper_bound(x,    -d,    u) === r
            end
        elseif d < zero(d)
            if isnan(l) # test as if unbounded below
                r = +Inf
                @test step_to_upper_bound(  x, +,  d,    u) === r
                @test step_to_upper_bound(  x,     d,    u) === r
                @test step_to_upper_bound(  x, -, -d,    u) === r
                @test step_from_upper_bound(x,    -d,    u) === r
            else
                r = l > typemin(l) ? (l - x)/d : +Inf
                @test step_to_bounds(       x, +,  d, l, u) === r
                @test step_to_bounds(       x,     d, l, u) === r
                @test step_to_bounds(       x, -, -d, l, u) === r
                @test step_from_bounds(     x,    -d, l, u) === r
                @test step_to_lower_bound(  x, +,  d, l   ) === r
                @test step_to_lower_bound(  x,     d, l   ) === r
                @test step_to_lower_bound(  x, -, -d, l   ) === r
                @test step_from_lower_bound(x,    -d, l   ) === r
            end
        elseif iszero(d)
            @test isnan(step_to_bounds(       x, +,  d, l, u))
            @test isnan(step_to_bounds(       x,     d, l, u))
            @test isnan(step_to_bounds(       x, -,  d, l, u))
            @test isnan(step_from_bounds(     x,     d, l, u))
            @test isnan(step_to_lower_bound(  x, +,  d, l   ))
            @test isnan(step_to_lower_bound(  x,     d, l   ))
            @test isnan(step_to_lower_bound(  x, -,  d, l   ))
            @test isnan(step_from_lower_bound(x,     d, l   ))
            @test isnan(step_to_upper_bound(  x, +,  d,    u))
            @test isnan(step_to_upper_bound(  x,     d,    u))
            @test isnan(step_to_upper_bound(  x, -,  d,    u))
            @test isnan(step_from_upper_bound(x,     d,    u))
        end
    end

    dims = (3,4)
    T = Float32 # bounds below are purposely in Float64
    N = length(dims)
    vals = -5:prod(dims)-6
    bounds = Dict(
        # Unconstrained cases:
        "(nothing,nothing)"    => (nothing, nothing),
        "(-Inf,nothing)"       => (-Inf, nothing),
        "(nothing,+Inf)"       => (nothing, +Inf),
        "(-Inf,nothing)"       => (-Inf, nothing),
        "(-Inf,+Inf)"          => (-Inf, +Inf),
        "([-Inf,...],+Inf)"    => (fill(-Inf,dims), +Inf),
        "(nothing,[+Inf,...])" => (-Inf, fill(+Inf,dims)),
        # Bounded below:
        "(0,nothing)"          => (0, nothing),
        "(0,+Inf)"             => (0, +Inf),
        "([0,...],+Inf)"       => (zeros(dims), +Inf),
        "([0,...],[+Inf,...])" => (zeros(dims), fill(+Inf,dims)),
        # Bounded above:
        "(nothing,0)"          => (nothing, 0),
        "(-Inf,0)"             => (-Inf, 0),
    )

    @testset "Simple operations on bounded sets ($B)" for B in keys(bounds)
        x0 = rand((zero(T),one(T)), dims)
        x1 = x0 .- T(1.1) # x1 < 0 holds somewhere
        x2 = x0 .+ T(1.1) # x2 > 1 holds somewhere
        l, u = bounds[B]
        below = l isa Nothing ? false :
                l isa Number ? l > -Inf :
                l isa AbstractUniformArray ? value(l) > -Inf : true
        @test is_bounding_below(l) === below
        above = u isa Nothing ? false :
                u isa Number ? u < +Inf :
                u isa AbstractUniformArray ? value(u) < +Inf : true
        @test is_bounding_above(u) === above
        L = l isa Nothing ? UniformArray{T}(-Inf, dims) :
            l isa Number  ? UniformArray{T}(l, dims) :
            convert_eltype(T, l)
        U = u isa Nothing ? UniformArray{T}(+Inf, dims) :
            u isa Number  ? UniformArray{T}(u, dims) :
            convert_eltype(T, u)
        Ω = @inferred BoundedSet(x0; lower=l, upper=u)
        P = @inferred Projector(Ω)
        @test @inferred(eltype(Ω)) <: AbstractArray
        @test @inferred(eltype(eltype(Ω))) === float(eltype(x0)) === T
        @test @inferred(ndims(eltype(Ω))) === ndims(x0) === N
        @test @inferred(eltype(Ω)) === AbstractArray{T,N}
        @test Ω.lower isa AbstractArray{T,N}
        @test Ω.upper isa AbstractArray{T,N}
        @test (Ω.lower isa AbstractUniformArray) == (l isa Union{Nothing,Number,AbstractUniformArray})
        @test (Ω.upper isa AbstractUniformArray) == (u isa Union{Nothing,Number,AbstractUniformArray})
        if Ω.lower isa AbstractUniformArray
            @test value(Ω.lower) === T(l === nothing ? -Inf : l isa Number ? l : value(l))
        end
        if Ω.upper isa AbstractUniformArray
            @test value(Ω.upper) === T(u === nothing ? +Inf : u isa Number ? u : value(u))
        end
        @test @inferred(first(Ω)) === Ω.lower
        @test @inferred(last(Ω)) === Ω.upper
        @test @inferred(length(Ω)) === 2
        @test (Ω...,) === (first(Ω), last(Ω))
        @test is_bounding(Ω) === (below, above)
        @test isempty(Ω) == mapreduce(>, |, L, U; init=false)
        if !below && !above
            # Unconstrained case.
            @test x0 ∈ Ω
            @test P(x0) == x0
            @test x1 ∈ Ω
            @test P(x1) == x1
            @test x2 ∈ Ω
            @test P(x2) == x2
        end
        @test Ω === @inferred BoundedSet(Ω...,)
        @test Ω === @inferred BoundedSet{eltype(eltype(Ω))}(Ω...,)
        @test Ω === @inferred BoundedSet{eltype(eltype(Ω)),ndims(eltype(Ω))}(Ω...,)
        #=
        Ω = @inferred BoundedSet(zeros(T,dims), nothing)
        P = @inferred Projector(Ω)
        @test !isempty(Ω)
        @test   x0 ∈ Ω
        @test !(x1 ∈ Ω)
        @test   x2 ∈ Ω
        Ω.lower[2] = NaN
        @test isempty(Ω)
        @test !(x0 ∈ Ω)
        @test !(x1 ∈ Ω)
        @test !(x2 ∈ Ω)
        Ω = @inferred BoundedSet{T,N}(nothing, ones(T,dims))
        @test !isempty(Ω)
        @test   x0 ∈ Ω
        @test   x1 ∈ Ω
        @test !(x2 ∈ Ω)
        Ω.upper[2] = NaN
        @test isempty(Ω)
        @test !(x0 ∈ Ω)
        @test !(x1 ∈ Ω)
        @test !(x2 ∈ Ω)
        Ω = @inferred BoundedSet{T,N}(zeros(T,dims), ones(T,dims))
        @test !isempty(Ω)
        @test   x0 ∈ Ω
        @test !(x1 ∈ Ω)
        @test !(x2 ∈ Ω)
        Ω.lower[2] = NaN
        @test isempty(Ω)
        @test !(x0 ∈ Ω)
        @test !(x1 ∈ Ω)
        @test !(x2 ∈ Ω)
        Ω.lower[2] = 0 # restore lower bound
        Ω.upper[2] = NaN
        @test isempty(Ω)
        @test !(x0 ∈ Ω)
        @test !(x1 ∈ Ω)
        @test !(x2 ∈ Ω)
        =#
    end

    #=
    @testset "Conversion of bounded sets (Ω = $B)" for B in keys(bounds)
        atol = 0
        rtol = 2eps(Float32)
        T₁ = Float64
        T₂ = Float32
        Ω₁ = @inferred BoundedSet{T₁,N}(bounds[B]...)
        Ω₂ = @inferred BoundedSet{T₂,N}(bounds[B]...)
        Ω₃ = @inferred BoundedSet{T₂,N}(Ω₁)
        Ω₄ = @inferred convert(BoundedSet{T₁,N}, Ω₂)
        @test BoundedSet{T₁,N}(Ω₁) === Ω₁
        @test BoundedSet{T₂,N}(Ω₂) === Ω₂
        @test convert(BoundedSet{T₁,N}, Ω₁) === Ω₁
        @test convert(BoundedSet{T₂,N}, Ω₂) === Ω₂
        @test Ω₃ isa BoundedSet{T₂,N}
        @test Ω₄ isa BoundedSet{T₁,N}
        if bounds[B][1] === nothing
            @test Ω₃.lower === Ω₂.lower === nothing
            @test Ω₄.lower === Ω₁.lower === nothing
        else
            @test Ω₃.lower ≈ Ω₂.lower atol=atol rtol=rtol
            @test Ω₄.lower ≈ Ω₁.lower atol=atol rtol=rtol
        end
        if bounds[B][2] === nothing
            @test Ω₃.upper === Ω₂.upper === nothing
            @test Ω₄.upper === Ω₁.upper === nothing
        else
            @test Ω₃.upper ≈ Ω₂.upper atol=atol rtol=rtol
            @test Ω₄.upper ≈ Ω₁.upper atol=atol rtol=rtol
        end
    end

    @testset "Project variables, etc. ($T, $(pm)d, Ω = $B)" for T in floats,
        pm in (+, -), B in keys(bounds)

        Ω = @inferred BoundedSet{T,N}(bounds[B]...)
        @test Ω isa BoundedSet{T,N}
        P = @inferred Projector(Ω)
        x0 = Array{T}(reshape(vals, dims))
        x = @inferred Generic.project_variables!(similar(x0), x0, Ω) # make sure x is feasible
        @test x isa Array{T,N}
        y = similar(x)
        z = similar(x)
        d = ones(T, size(x))
        if pm === +
            @test y === @inferred project_variables!(y, x0, Ω)
            @test y == x
            @test x == @inferred P(x0)
            @test y === @inferred P(y, x0)
            @test y == x
        end
        @test y === @inferred project_direction!(y, x, pm, d, Ω)
        @test y == @inferred Generic.project_direction!(z, x, pm, d, Ω)
        @test y === @inferred changing_variables!(y, x, pm, d, Ω)
        @test y == @inferred Generic.changing_variables!(z, x, pm, d, Ω)
        amin, amax = @inferred linesearch_limits(x, pm, d, Ω)
        @test  amin        == @inferred linesearch_stepmin(x, pm, d, Ω)
        @test        amax  == @inferred linesearch_stepmax(x, pm, d, Ω)
        @test (amin, amax) == @inferred linesearch_limits(x, pm, d, Ω)
    end
    =#

end
nothing
