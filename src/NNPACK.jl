module NNPACK

using Libdl, Requires

include("libnnpack_types.jl")
include("error.jl")
include("libnnpack.jl")
include("utils.jl")

export conv, ∇conv_data, ∇conv_filter, softmax, maxpool

const depsjl_path = joinpath(dirname(@__FILE__), "..", "deps", "deps.jl")
if !isfile(depsjl_path)
    error("NNPACK not installed properly, run Pkg.build(\"NNPACK\"), restart Julia and try again")
end
include(depsjl_path)

const interface_path = joinpath(dirname(@__FILE__), "interface.jl")
const full_interface_path = joinpath(dirname(@__FILE__), "interface_full.jl")
global shared_threadpool = Ref(C_NULL)

@init begin
    check_deps()
    status = nnp_initialize()
    status == nnp_status_unsupported_hardware && error("HARDWARE is unsupported by NNPACK")
    try
        global NNPACK_CPU_THREADS = parse(UInt64, ENV["NNPACK_CPU_THREADS"])
    catch
        global NNPACK_CPU_THREADS = 4
    end
    include(interface_path)
    try
        global NNPACK_FAST_OPS = parse(UInt64, ENV["NNPACK_FAST_OPS"])
    catch
        global NNPACK_FAST_OPS = 1
    end
    NNPACK_FAST_OPS == 1 && include(full_interface_path)
    global shared_threadpool = Ref(pthreadpool_create(NNPACK_CPU_THREADS))
end

end
