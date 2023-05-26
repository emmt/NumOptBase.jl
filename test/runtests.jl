module TestingNumOptBase

using NumOptBase
using Test
using Base: @propagate_inbounds

# Reference methods.
function ref_norm1(x)
    s = zero(eltype(x))
    for xᵢ in x
        s += abs(xᵢ)
    end
    return s
end
function ref_norm2(x)
    s = zero(eltype(x))
    for xᵢ in x
        s += abs2(xᵢ)
    end
    return sqrt(s)
end
function ref_norminf(x)
    s = zero(eltype(x))
    for xᵢ in x
        s = max(s, abs(xᵢ))
    end
    return s
end
function ref_inner(x,y)
    s = zero(eltype(x))*zero(eltype(y))
    for (xᵢ,yᵢ) in zip(x,y)
        s += xᵢ*yᵢ
    end
    return s
end
function ref_inner(w,x,y)
    s = zero(eltype(w))*zero(eltype(x))*zero(eltype(y))
    for (wᵢ,xᵢ,yᵢ) in zip(w,x,y)
        s += wᵢ*xᵢ*yᵢ
    end
    return s
end

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
    T′s = (Float32, Float64)
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
                @test typeof(res) === T
                @test res ≈ ref_norm1(x)
            end
        end
        @testset "norm2" begin
            let res = @inferred NumOptBase.norm2(x)
                @test typeof(res) === T
                @test res ≈ ref_norm2(x)
            end
        end
        @testset "norminf" begin
            let res = @inferred NumOptBase.norminf(x)
                @test typeof(res) === T
                @test res ≈ ref_norminf(x)
            end
        end
        @testset "inner product" begin
            let res = @inferred NumOptBase.inner(x, y)
                @test typeof(res) === T
                @test res ≈ ref_inner(x, y)
            end
            let res = @inferred NumOptBase.inner(w, x, y)
                @test typeof(res) === T
                @test res ≈ ref_inner(w, x, y)
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
