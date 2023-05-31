module TestingNumOptBase

using NumOptBase
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
@inline Base.axes(A::MyArray) = axes(A.parent)
@inline Base.IndexStyle(::Type{<:MyArray{T,N,false}}) where {T,N} = IndexCartesian()
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

function runtests()
    T′s = (Float64, Complex{Float32})
    F′s = (:StridedArray, :AbstractArray)
    α′s = (0, 1, -1, 2.1)
    β′s = (0, 1, -1, -1.7)
    dims = (3,4,5)
    @testset "NumOptBase T=$T, $F" for F ∈ F′s, T ∈ T′s
        wrapper = F === :StridedArray ? identity : MyArray
        w = wrapper(rand(T, dims))
        x = wrapper(rand(T, dims))
        y = wrapper(rand(T, dims))
        z = similar(x)
        @testset "norm1" begin
            let res = @inferred NumOptBase.norm1(x)
                @test typeof(res) === real(T)
                @test res ≈ ref_norm1(x)
            end
        end
        @testset "norm2" begin
            let res = @inferred NumOptBase.norm2(x)
                @test typeof(res) === real(T)
                @test res ≈ ref_norm2(x)
            end
        end
        @testset "norminf" begin
            let res = @inferred NumOptBase.norminf(x)
                @test typeof(res) === real(T)
                @test res ≈ ref_norminf(x)
            end
        end
        @testset "inner product" begin
            let res = @inferred NumOptBase.inner(x, y)
                @test typeof(res) === real(T)
                @test res ≈ ref_inner(x, y)
            end
            if T <: Complex
                @test_throws Exception NumOptBase.inner(w, x, y)
            else
                let res = @inferred NumOptBase.inner(w, x, y)
                    @test typeof(res) === real(T)
                    @test res ≈ ref_inner(w, x, y)
                end
            end
        end
        @testset "scale! α = $α" for α in α′s
            let res = @inferred NumOptBase.scale!(z, α, x)
                @test res === z
                @test res ≈ (@. α*x)
            end
        end
        @testset "multiply!" begin
            let res = @inferred NumOptBase.multiply!(z, x, y)
                @test res === z
                @test res ≈ (@. x*y)
            end
        end
        @testset "update! α = $α" for α in α′s
            let res = @inferred NumOptBase.update!(copyto!(z, y), α, x)
                @test res === z
                @test res ≈ (@. y + α*x)
            end
        end
        @testset "combine! α = $α, β = $β" for α in α′s, β ∈ β′s
            let res = @inferred NumOptBase.combine!(z, α, x, β, y)
                @test res === z
                @test res ≈ (@. α*x + β*y)
            end
        end
    end
    nothing
end

end # module

println(stderr, "Testing NumOptBase package...")
TestingNumOptBase.runtests()
if !isdefined(Main,:LoopVectorization)
    println(stderr, "\nTesting NumOptBase package with LoopVectorization...")
    using LoopVectorization
    TestingNumOptBase.runtests()
end
