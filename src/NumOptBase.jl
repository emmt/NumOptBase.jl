"""

Package `NumOptBase` provides basic operations on "variables" for numerical
optimization methods.

"""
module NumOptBase

using TypeUtils
using ArrayTools: @assert_same_axes
using LinearAlgebra

if !isdefined(Base, :get_extension)
    using Requires
end

include("private.jl")
include("public.jl")

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include(
            "../ext/NumOptBaseCUDAExt.jl")
        @require LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890" include(
            "../ext/NumOptBaseLoopVectorizationExt.jl")
    end
end

end # module
