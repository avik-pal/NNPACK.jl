module NNPACK

export relu, leaky_relu, ∇relu, ∇leaky_relu

include("libnnpack_helpers.jl")
include("error.jl")
include("libnnpack.jl")
include("interface.jl")

__init__() = nnp_initialize()

end
