module NNPACK

export relu, leaky_relu, ∇relu, ∇leaky_relu

include("libnnpack_helpers.jl")
include("libnnpack.jl")
include("error.jl")
include("interface.jl")

__init__() = nnp_initialize()

end
