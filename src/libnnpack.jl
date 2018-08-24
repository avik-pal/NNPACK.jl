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
    nnp_relu_output(prod(size(x)[N-1:N]), prod(size(x)[1:N-2]), x, y, negative_slope, threadpool)
    y
end

function nnp_relu_input_gradient(batch_size, channels, grad_output, input, grad_input, negative_slope, threadpool)
    @check ccall((:nnp_relu_input_gradient, "libnnpack"), nnp_status, (Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cfloat, pthreadpool_t), batch_size, channels, grad_output, input, grad_input, negative_slope, threadpool)
end

function nnp_relu_input_gradient(x::AbstractArray{Float32,N}, dy::AbstractArray{Float32,N}; negative_slope::AbstractFloat = 0.0, threadpool = nothing) where {N}
    dx = zeros(Float32, size(x))
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    # Investigate why the channel and batch dims need to specified like this
    nnp_relu_input_gradient(Csize_t(prod(size(x)[N-1:N])), prod(size(x)[1:N-2]), dy, x, dx, negative_slope, threadpool)
    dx
end

function nnp_softmax_output(batch_size, channels, input, output, threadpool)
    @check ccall((:nnp_softmax_output, "libnnpack"), nnp_status, (Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, pthreadpool_t), batch_size, channels, input, output, threadpool)
end

function nnp_softmax_output(x::AbstractVecOrMat{Float32}; inplace::Bool = true, threadpool = nothing)
    y = inplace ? x : zeros(Float32, size(x))
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    nnp_softmax_output(ndims(x) == 2 ? size(x, 2) : 1, size(x, 1), x, y, threadpool)
    y
end

function nnp_fully_connected_output(batch_size, input_channels, output_channels, input, kernel, output, threadpool, profile)
    @check ccall((:nnp_fully_connected_output, "libnnpack"), nnp_status, (Csize_t, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, pthreadpool_t, Ptr{Nothing}), batch_size, input_channels, output_channels, input, kernel, output, threadpool, C_NULL)
end

function nnp_fully_connected_output(x::AbstractArray{Float32,2}, w::AbstractArray{Float32,2}; profile = nothing, threadpool = nothing)
    input_channels, batch_size = size(x)
    output_channels = size(w, 1)
    y = zeros(Float32, output_channels, batch_size)
    profile = profile == nothing ? nnp_profile() : profile
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    nnp_fully_connected_output(batch_size, input_channels, output_channels, x, w, y, threadpool, profile)
    y
end

function nnp_fully_connected_inference_f16f32(input_channels, output_channels, input, kernel, output, threadpool)
    @check ccall((:nnp_fully_connected_inference_f16f32, "libnnpack"), nnp_status, (Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Nothing}, Ptr{Cfloat}, pthreadpool_t), input_channels, output_channels, input, kernel, output, threadpool)
end

nnp_fully_connected_inference_f16f32(x::AbstractVector{Float32}, w::AbstractArray{Float16,2}; threadpool = nothing) =
    nnp_fully_connected_inference(reshape(x, size(x), 1), w, threadpool = threadpool)

function nnp_fully_connected_inference_f16f32(x::AbstractMatrix{Float32}, w::AbstractArray{Float16,2}; threadpool = nothing)
    input_channels = size(x, 1)
    output_channels = size(x, 1)
    y = zeros(Float32, output_channels, 1)
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    nnp_fully_connected_inference(input_channels, output_channels, x, w, y, threadpool)
    y
end

function nnp_fully_connected_inference(input_channels, output_channels, input, kernel, output, threadpool)
    @check ccall((:nnp_fully_connected_inference, "libnnpack"), nnp_status, (Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, pthreadpool_t), input_channels, output_channels, input, kernel, output, threadpool)
end

nnp_fully_connected_inference(x::AbstractVector{Float32}, w::AbstractArray{Float32,2}; threadpool = nothing) =
    nnp_fully_connected_inference(reshape(x, size(x), 1), w, threadpool = threadpool)

function nnp_fully_connected_inference(x::AbstractMatrix{Float32}, w::AbstractArray{Float32,2}; threadpool = nothing)
    input_channels = size(x, 1)
    output_channels = size(x, 1)
    y = zeros(Float32, output_channels, 1)
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    nnp_fully_connected_inference(input_channels, output_channels, x, w, y, threadpool)
    y
end
