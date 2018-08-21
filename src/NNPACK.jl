module NNPACK

include("libnnpack_helpers.jl")
include("libnnpack.jl")

__init__() = nnp_initialize()

end
