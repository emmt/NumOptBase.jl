using Test

if !isdefined(@__MODULE__, :TestingNumOptBase)
    include("testing.jl")
end

@testset "NumOptBase package" begin
    TestingNumOptBase.test_utilities()
    TestingNumOptBase.test_operations()
    TestingNumOptBase.test_operators()

    if !isdefined(Main,:LoopVectorization)
        using LoopVectorization
        println()
        @testset "Using LoopVectorization" begin
            TestingNumOptBase.test_operations()
        end
    end
end

# To test with CUDA, do something like:
# @testset "NumOptBase package with CUDA" begin
#     TestingNumOptBase.test_operationss(; classes = (CuArray,))
# end

nothing
