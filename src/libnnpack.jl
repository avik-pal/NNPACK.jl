function nnp_initialize()
	ccall((:nnp_initialize, "libnnpack"), nnp_status, (), )
end

function nnp_deinitialize()
	ccall((:nnp_deinitialize, "libnnpack"), nnp_status, (), )
end

function pthreadpool_create(n::Int = 0)
	ccall((:pthreadpool_create, "libnnpack"), Ptr{Nothing}, (Csize_t), n)
end
