module NNPACK

export relu, leaky_relu, ∇relu, ∇leaky_relu,
       softmax, fully_connected

include("libnnpack_types.jl")
include("error.jl")
include("libnnpack.jl")
include("interface.jl")

__init__() = nnp_initialize()

end
