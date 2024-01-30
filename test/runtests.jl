using NumOptBase, Test, LinearAlgebra

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
        engine = NumOptBase.engine
        only_arrays = NumOptBase.only_arrays
        check_axes = NumOptBase.check_axes
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
        @test nothing === @inferred check_axes()
        @test nothing === @inferred check_axes(x)
        @test nothing === @inferred check_axes(x, y)
        @test_throws DimensionMismatch check_axes(w, x)
        @test_throws DimensionMismatch check_axes(x, x')
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
        @testset "norm1" begin
            let xᵢ = first(x_ref), res = @inferred norm1(xᵢ)
                @test typeof(res) === real(T)
                @test res ≈ Generic.norm1(xᵢ)
            end
            let res = @inferred norm1(x)
                @test typeof(res) === real(T)
                @test res ≈ Generic.norm1(x_ref)
            end
        end
        @testset "norm2" begin
            let xᵢ = first(x_ref), res = @inferred norm2(xᵢ)
                @test typeof(res) === real(T)
                @test res ≈ Generic.norm2(xᵢ)
            end
            let res = @inferred norm2(x)
                @test typeof(res) === real(T)
                @test res ≈ Generic.norm2(x_ref)
            end
        end
        @testset "norminf" begin
            let xᵢ = first(x_ref), res = @inferred norminf(xᵢ)
                @test typeof(res) === real(T)
                @test res ≈ Generic.norminf(xᵢ)
            end
            let res = @inferred norminf(x)
                @test typeof(res) === real(T)
                @test res ≈ Generic.norminf(x_ref)
            end
        end
        @testset "inner product" begin
            let res = @inferred inner(x, y)
                @test typeof(res) === real(T)
                @test res ≈ Generic.inner(x_ref, y_ref)
            end
            if T <: Complex
                @test_throws Exception inner(w, x, y)
            else
                let res = @inferred inner(w, x, y)
                    @test typeof(res) === real(T)
                    @test res ≈ Generic.inner(w_ref, x_ref, y_ref)
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
    end

    @testset "Array Operators ($A{$T,$(length(dims))})" for A in array_types, T ∈ floats
        Diag = NumOptBase.Diag
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
    end

    dims = (3,4)
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
        @test y === @inferred unblocked_variables!(y, x, pm, d, Ω)
        @test y == @inferred Generic.unblocked_variables!(z, x, pm, d, Ω)
        amin, amax = @inferred linesearch_limits(x, pm, d, Ω)
        @test  amin        == @inferred linesearch_stepmin(x, pm, d, Ω)
        @test        amax  == @inferred linesearch_stepmax(x, pm, d, Ω)
        @test (amin, amax) == @inferred linesearch_limits(x, pm, d, Ω)
    end

end
nothing
