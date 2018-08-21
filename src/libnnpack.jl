function nnp_initialize()
	ccall((:nnp_initialize, "libnnpack"), nnp_status, (),)
end

function nnp_deinitialize()
	ccall((:nnp_deinitialize, "libnnpack"), nnp_status, (),)
end

function pthreadpool_create(n::Int = 0)
	ccall((:pthreadpool_create, "libnnpack"), Ptr{Nothing}, (Csize_t,), n)
end

function nnp_relu_output(batch_size, channels, input, output, negative_slope, threadpool)
	@check ccall((:nnp_relu_output, "libnnpack"), nnp_status, (Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Cfloat, pthreadpool_t), batch_size, channels, input, output, negative_slope, threadpool)
end

function nnp_relu_output(x::AbstractArray{Float32,N}; inplace::Bool = true, negative_slope::AbstractFloat = 0.0, threadpool = nothing) where {N}
	y = inplace ? x : zeros(Float32, size(x))
	threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
	# Investigate why the channel and batch dims need to specified like this
	nnp_relu_output(Csize_t(prod(size(x)[N-1:N])), Csize_t(prod(size(x)[1:N-2])), x, y, Cfloat(negative_slope), threadpool)
	y
end
