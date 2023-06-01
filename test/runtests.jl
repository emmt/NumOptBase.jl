using Test

if !isdefined(@__MODULE__, :TestingNumOptBase)
    include("testing.jl")
end

@testset "NumOptBase package" TestingNumOptBase.runtests()

if !isdefined(Main,:LoopVectorization)
    using LoopVectorization
    println()
    @testset "NumOptBase package with LoopVectorization" TestingNumOptBase.runtests()
end

# To test with CUDA, do something like:
# @testset "NumOptBase package with CUDA" TestingNumOptBase.runtests(; classes = (CuArray,))

nothing
