using Test

if !isdefined(@__MODULE__, :TestingNumOptBase)
    include("testing.jl")
end

TestingNumOptBase.test_all()

if !isdefined(Main,:LoopVectorization)
    using LoopVectorization
    println()
    @testset "... with LoopVectorization" begin
        TestingNumOptBase.test_operations()
    end
end

if isdefined(Main,:CUDA)
    @testset "NumOptBase package with CUDA" begin
        TestingNumOptBase.test_operations(; classes = (CuArray,))
    end
end

nothing
