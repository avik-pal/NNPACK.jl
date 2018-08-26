module NNPACK

export relu, leaky_relu, ∇relu, ∇leaky_relu,
       softmax, fully_connected, maxpool2d

include("libnnpack_types.jl")
include("error.jl")
include("utils.jl")
include("libnnpack.jl")
include("interface.jl")

__init__() = nnp_initialize()

end
