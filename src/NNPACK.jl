__precompile__()

module NNPACK

using Libdl, Requires

include("libnnpack_types.jl")
include("error.jl")
include("libnnpack.jl")

const depsjl_path = joinpath(dirname(@__FILE__), "..", "deps", "deps.jl")
if !isfile(depsjl_path)
    error("NNPACK not installed properly, run Pkg.build(\"NNPACK\"), restart Julia and try again")
end
include(depsjl_path)

function __init__()
    check_deps()
    nnp_initialize()
    has_nnlib = false
    @require NNlib="872c559c-99b0-510c-b3b7-b6c96a88d5cd" has_nnlib=true
    if has_nnlib
        include(joinpath(dirname(@__FILE__), "nnlib.jl"))
    else
        include(joinpath(dirname(@__FILE__), "interface.jl"))
        include(joinpath(dirname(@__FILE__), "utils.jl"))
    end
end

end
