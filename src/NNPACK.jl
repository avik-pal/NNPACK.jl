module NNPACK

using Libdl

export relu, leaky_relu, ∇relu, ∇leaky_relu,
       softmax, fully_connected, maxpool2d

include("libnnpack_types.jl")
include("error.jl")
include("utils.jl")
include("libnnpack.jl")
include("interface.jl")

const depsjl_path = joinpath(dirname(@__FILE__), "..", "deps", "deps.jl")
if !isfile(depsjl_path)
    error("NNPACK not installed properly, run Pkg.build(\"NNPACK\"), restart Julia and try again")
end
include(depsjl_path)

function __init__()
    check_deps()
    nnp_initialize()
end

end
