module NNPACK

using Libdl, Requires

include("libnnpack_types.jl")
include("error.jl")
include("libnnpack.jl")

has_nnlib = false

const depsjl_path = joinpath(dirname(@__FILE__), "..", "deps", "deps.jl")
if !isfile(depsjl_path)
    error("NNPACK not installed properly, run Pkg.build(\"NNPACK\"), restart Julia and try again")
end
include(depsjl_path)

const nnlib_interface_path = joinpath(dirname(@__FILE__), "interface.jl")
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
    global shared_threadpool = Ref(pthreadpool_create(NNPACK_CPU_THREADS))
end

end
