"""

Package `NumOptBase` provides basic operations on "variables" for numerical
optimization methods.

"""
module NumOptBase

export
    # Vectorized operations:
    apply!,
    combine!,
    # copy!, # FIXME: has a different semantic in Julia Base
    diag,
    inner,
    multiply!,
    norm1,
    norm2,
    norminf,
    scale!,
    update!,
    zerofill!

export
    # Bound constraints:
    Bound,
    BoundedSet,
    linesearch_limits,
    linesearch_stepmax,
    linesearch_stepmin,
    project_direction!,
    project_variables!,
    unblocked_variables!

using TypeUtils
using ArrayTools: @assert_same_axes
using StructuredArrays
using LinearAlgebra

if !isdefined(Base, :get_extension)
    using Requires
end

include("types.jl")
include("utils.jl")
include("vops.jl")
include("bounds.jl")

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include(
            "../ext/NumOptBaseCUDAExt.jl")
        @require LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890" include(
            "../ext/NumOptBaseLoopVectorizationExt.jl")
    end
end

end # module
