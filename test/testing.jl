module TestingNumOptBase

using NumOptBase
using LinearAlgebra
using Test
using Base: @propagate_inbounds

# Reference methods (NOTE: muti-dimensional arrays are treated as vectors and
# complexes as pairs of reals).
ref_norm1(x::Real) = abs(x)
ref_norm1(x::Complex) = ref_norm1(real(x)) + ref_norm1(imag(x))
ref_norm1(x::AbstractArray) =
    mapreduce(ref_norm1, +, x; init = ref_norm1(zero(eltype(x))))

ref_norm2(x::Real) = abs(x)
ref_norm2(x::Complex) = sqrt(abs2(x))
ref_norm2(x::AbstractArray) =
    sqrt(mapreduce(abs2, +, x; init = abs2(zero(eltype(x)))))

ref_norminf(x::Real) = abs(x)
ref_norminf(x::Complex) = max(abs(real(x)), abs(imag(x)))
ref_norminf(x::AbstractArray) =
    mapreduce(ref_norminf, max, x; init = ref_norminf(zero(eltype(x))))

ref_inner(x::Real, y::Real) = x*y
ref_inner(x::Complex, y::Complex) = real(x)*real(y) + imag(x)*imag(y)
ref_inner(w::Real, x::Real, y::Real) = w*x*y
ref_inner(x::AbstractArray, y::AbstractArray) =
    mapreduce(ref_inner, +, x, y; init = ref_inner(zero(eltype(x)), zero(eltype(y))))
ref_inner(w::AbstractArray, x::AbstractArray, y::AbstractArray) =
    mapreduce(ref_inner, +, w, x, y; init = ref_inner(zero(eltype(w)), zero(eltype(x)), zero(eltype(y))))

# Custom array type to check other versions than the ones that apply for
# strided arrays.
struct MyArray{T,N,L,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    MyArray(arr::A) where {T,N,A<:AbstractArray{T,N}} =
        new{T,N,IndexStyle(A)===IndexLinear(),A}(arr)
end
Base.parent(A::MyArray) = A.parent
Base.length(A::MyArray) = length(A.parent)
Base.size(A::MyArray) = size(A.parent)
Base.axes(A::MyArray) = axes(A.parent)
Base.IndexStyle(::Type{<:MyArray{T,N,false}}) where {T,N} = IndexCartesian()
@inline function Base.getindex(A::MyArray{T,N,false}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    return @inbounds getindex(parent(A), I...)
end
@inline function Base.setindex!(A::MyArray{T,N,false}, x, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    @inbounds setindex!(parent(A), x, I...)
    return A
end
@inline Base.IndexStyle(::Type{<:MyArray{T,N,true}}) where {T,N} = IndexLinear()
@inline function Base.getindex(A::MyArray{T,N,true}, i::Int) where {T,N}
    @boundscheck checkbounds(A, i)
    return @inbounds getindex(parent(A), i)
end
@inline function Base.setindex!(A::MyArray{T,N,true}, x, i::Int) where {T,N}
    @boundscheck checkbounds(A, i)
    @inbounds setindex!(parent(A), x, i)
    return A
end
Base.similar(A::MyArray, ::Type{T}, dims::Dims) where {T} =
    MyArray(similar(parent(A), T, dims))

function test_utilities()
    @testset "Utilities" begin
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

        only_arrays = NumOptBase.only_arrays
        check_axes = NumOptBase.check_axes
        w = collect(-1:4)
        x = zeros(Float64, 2,3)
        y = ones(Float32, size(x))
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
end

function test_operations(;
                         classes = (:StridedArray, :AbstractArray),
                         eltypes = (Float64, Complex{Float32}),
                         dims = (3,4,5),
                         alphas = (0, 1, -1, 2.1),
                         betas = (0, 1, -1, -1.7))
    @testset "$F{$T,$(length(dims))}" for F ∈ classes, T ∈ eltypes
        wrapper =
            (F === :StridedArray || F === StridedArray) ? identity :
            (F === :AbstractArray || F === AbstractArray) ? MyArray :
            (F isa UnionAll || F isa DataType) ? F :
            error("unexpected array class $F")
        w_ref = rand(T, dims)
        x_ref = rand(T, dims)
        y_ref = rand(T, dims)
        tmp = Array{T}(undef, dims)
        w = wrapper(w_ref)
        x = wrapper(x_ref)
        y = wrapper(y_ref)
        z = similar(x)
        @testset "norm1" begin
            let xᵢ = first(x_ref), res = @inferred norm1(xᵢ)
                @test typeof(res) === real(T)
                @test res ≈ ref_norm1(xᵢ)
            end
            let res = @inferred norm1(x)
                @test typeof(res) === real(T)
                @test res ≈ ref_norm1(x_ref)
            end
        end
        @testset "norm2" begin
            let xᵢ = first(x_ref), res = @inferred norm2(xᵢ)
                @test typeof(res) === real(T)
                @test res ≈ ref_norm2(xᵢ)
            end
            let res = @inferred norm2(x)
                @test typeof(res) === real(T)
                @test res ≈ ref_norm2(x_ref)
            end
        end
        @testset "norminf" begin
            let xᵢ = first(x_ref), res = @inferred norminf(xᵢ)
                @test typeof(res) === real(T)
                @test res ≈ ref_norminf(xᵢ)
            end
            let res = @inferred norminf(x)
                @test typeof(res) === real(T)
                @test res ≈ ref_norminf(x_ref)
            end
        end
        @testset "inner product" begin
            let res = @inferred inner(x, y)
                @test typeof(res) === real(T)
                @test res ≈ ref_inner(x_ref, y_ref)
            end
            if T <: Complex
                @test_throws Exception inner(w, x, y)
            else
                let res = @inferred inner(w, x, y)
                    @test typeof(res) === real(T)
                    @test res ≈ ref_inner(w_ref, x_ref, y_ref)
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
end

function test_operators()
    Diag = NumOptBase.Diag
    @testset "Operators" begin
        T = Float32
        dims = (2,3,4)
        N = length(dims)
        w = rand(T, dims)
        x = rand(T, dims)
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
end

"""
    same_float(x, y) -> bool

yields whether `x` and `y` have the same floating-point values. NaNs of any
kind are considered as the same floating-point value. Plus zero and minus zero
are considered as different floating-point values. Because of the former
assumption, `x === y` is not sufficient to test whether `x` and `y` have the
same floating-point values.

"""
same_float(x, y) = false
same_float(x::T, y::T) where {T <: AbstractFloat} =
    # Comparing results is tricky because not all NaNs are the same.
    (isnan(x) & isnan(y)) | (x === y)
function same_float(x::AbstractArray{T,N},
                    y::AbstractArray{T,N}; verb::Bool = false) where {T,N}
    @assert axes(x) == axes(y)
    result = true
    for i in CartesianIndices(x)
        flag = same_float(x[i], y[i])
        result &= flag
        if verb && !flag
            printstyled(stderr, "ERROR: $(x[i]) ≠ $(y[i]) at indices $(Tuple(i))\n";
                        color = :red)
        end
    end
    return result
end

lower_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L<:Nothing,U} = typemin(T)
lower_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L<:Number,U} = Ω.lower
lower_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L<:AbstractArray,U} = Ω.lower[i]

upper_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L,U<:Nothing} = typemax(T)
upper_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L,U<:Number} = Ω.upper
upper_bound(Ω::BoundedSet{T,N,L,U}, i) where {T,N,L,U<:AbstractArray} = Ω.upper[i]

# Reference version of project_variables!. Not meant to be smart, just to
# provide correct result.
function ref_project_variables!(dst::AbstractArray{T,N},
                                x::AbstractArray{T,N},
                                Ω::BoundedSet{T,N}) where {T,N}
   for i in eachindex(dst, x, NumOptBase.only_arrays(Ω.lower, Ω.upper)...)
        dst[i] = min(max(x[i], lower_bound(Ω, i)), upper_bound(Ω, i))
    end
    return dst
end

# Reference version of project_direction!. Not meant to be smart, just to
# provide correct result.
function ref_project_direction!(dst::AbstractArray{T,N},
                                x::AbstractArray{T,N},
                                pm::NumOptBase.PlusOrMinus, d::AbstractArray{T,N},
                                Ω::BoundedSet{T,N}) where {T,N}
    for i in eachindex(x, d, NumOptBase.only_arrays(Ω.lower, Ω.upper)...)
        s = pm(d[i])
        unblocked =
            s < zero(s) ? lower_bound(Ω, i) < x[i] :
            s > zero(s) ? upper_bound(Ω, i) > x[i] : true
        dst[i] = unblocked ? d[i] : zero(T)
    end
    return dst
end

# Reference version of unblocked_variables!. Not meant to be smart, just to
# provide correct result.
function ref_unblocked_variables!(dst::AbstractArray{B,N},
                                  x::AbstractArray{T,N},
                                  pm::NumOptBase.PlusOrMinus, d::AbstractArray{T,N},
                                  Ω::BoundedSet{T,N}) where {B,T,N}
    for i in eachindex(x, d, NumOptBase.only_arrays(Ω.lower, Ω.upper)...)
        s = pm(d[i])
        unblocked =
            s < zero(s) ? lower_bound(Ω, i) < x[i] :
            s > zero(s) ? upper_bound(Ω, i) > x[i] : true
        dst[i] = unblocked ? one(B) : zero(B)
    end
    return dst
end

# Reference version of linesearch_limits. Not meant to be smart, just to
# provide correct result.
function ref_linesearch_limits(x::AbstractArray{T,N},
                               pm::NumOptBase.PlusOrMinus, d::AbstractArray{T,N},
                               Ω::BoundedSet{T,N}) where {T,N}
    amin = ref_linesearch_stepmin(x, pm, d, Ω)
    amax = ref_linesearch_stepmax(x, pm, d, Ω)
    return amin, amax
end

# Reference version of linesearch_stepmin. Not meant to be smart, just to
# provide correct result.
function ref_linesearch_stepmin(x::AbstractArray{T,N},
                                 pm::NumOptBase.PlusOrMinus, d::AbstractArray{T,N},
                                 Ω::BoundedSet{T,N}) where {T,N}
    cnt = 0
    amin = typemax(T)
    for i in eachindex(x, d, NumOptBase.only_arrays(Ω.lower, Ω.upper)...)
        s = pm(d[i])
        if s < zero(s)
            l = lower_bound(Ω, i)
            if l > typemin(l)
                amin = min(amin, (l - x[i])/s)
            end
        elseif s > zero(s)
            u = upper_bound(Ω, i)
            if u < typemax(u)
                amin = min(amin, (u - x[i])/s)
            end
        end
    end
    return amin::T
end

# Reference version of linesearch_stepmax. Not meant to be smart, just to
# provide correct result.
function ref_linesearch_stepmax(x::AbstractArray{T,N},
                                 pm::NumOptBase.PlusOrMinus, d::AbstractArray{T,N},
                                 Ω::BoundedSet{T,N}) where {T,N}
    amax = T(NaN)
    for i in eachindex(x, d, NumOptBase.only_arrays(Ω.lower, Ω.upper)...)
        s = pm(d[i])
        if s < zero(s)
            l = lower_bound(Ω, i)
            if l > typemin(l)
                a = (l - x[i])/s
                if isnan(amax) || zero(a) ≤ a < amax
                    amax = a
                end
            end
        elseif s > zero(s)
            u = upper_bound(Ω, i)
            if u < typemax(u)
                a = (u - x[i])/s
                if isnan(amax) || zero(a) ≤ a < amax
                    amax = a
                end
            end
        end
    end
    return (isnan(amax) ? T(Inf) : amax)::T
end

function test_bounds()
    @testset "Bound constraints" begin
        @testset "Comparisons ($T)" for T in (Float32, Float64)
            @test  same_float(+zero(T), zero(T))
            @test !same_float(-zero(T), zero(T))
            @test  same_float(T(NaN), T(NaN))
            @test  same_float((+zero(T))/zero(T), T(NaN))
            @test  same_float((-zero(T))/zero(T), T(NaN))
            @test  same_float(T(+Inf), typemax(T))
            @test  same_float(T(-Inf), typemin(T))
            @test  same_float(one(T)/(+zero(T)), T(+Inf))
            @test  same_float(one(T)/(-zero(T)), T(-Inf))
            @test  same_float(zero(T), T(0))
            @test  same_float(one(T), T(1))

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
            @test same_float(r, s; verb=true)
            if isdefined(Main, :CUDA)
                vp = Main.CUDA.CuArray(v)
                r = Array(vp' ./ vp)
                @test eltype(r) === eltype(s) === T
                @test same_float(r, s; verb=true)
            end
        end

        dims = (3, 4)
        N = length(dims)
        vals = -5:6
        floats = (Float32, Float64)
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

        @testset "Project variables ($T, $(pm)d, Ω = $B)" for T in floats,
            pm in (+, -), B in keys(bounds)

            Ω = @inferred BoundedSet{T,N}(bounds[B]...)
            P = @inferred Projector(Ω)
            x0 = Array{T}(reshape(vals, dims))
            x = ref_project_variables!(similar(x0), x0, Ω) # make sure x is feasible
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
            @test y == ref_project_direction!(z, x, pm, d, Ω)
            @test y === @inferred unblocked_variables!(y, x, pm, d, Ω)
            @test y == ref_unblocked_variables!(z, x, pm, d, Ω)
            amin, amax = linesearch_limits(x, pm, d, Ω)
            @test  amin        == @inferred linesearch_stepmin(x, pm, d, Ω)
            @test        amax  == @inferred linesearch_stepmax(x, pm, d, Ω)
            @test (amin, amax) == @inferred linesearch_limits(x, pm, d, Ω)
        end
    end
end

# Run all tests with default settings.
function test_all()
    @testset "NumOptBase package" begin
        TestingNumOptBase.test_utilities()
        TestingNumOptBase.test_operations()
        TestingNumOptBase.test_operators()
        TestingNumOptBase.test_bounds()
    end
end

end # module
