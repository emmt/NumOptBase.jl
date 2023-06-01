using Test

if !isdefined(@__MODULE__, :TestingNumOptBase)
    include("testing.jl")
end

@testset "NumOptBase package" begin
    TestingNumOptBase.runtests()
end

if !isdefined(Main,:LoopVectorization)
    using LoopVectorization
    println()
    @testset "NumOptBase package with LoopVectorization" begin
        TestingNumOptBase.runtests()
    end
end

# To test with CUDA, do something like:
# @testset "NumOptBase package with CUDA" begin
#     TestingNumOptBase.runtests(; classes = (CuArray,))
# end

nothing
