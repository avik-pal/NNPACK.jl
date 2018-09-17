export relu, leakyrelu, ∇relu, ∇leakyrelu, softmax, fullyconnected,
       maxpool, conv, ∇conv_data, ∇conv_filter

function relu(x::AbstractArray{T,N}; inplace::Bool = true, nthreads::UInt64 = NNPACK_CPU_THREADS) where {T,N}
    T == Float32 || error("NNPACK RELU supports only Float32")
    N != 1 || error("NNPACK RELU requires an array of 2 or more dimensions")
    nnp_relu_output(x, inplace ? x : similar(x), threadpool = pthreadpool_create(nthreads))
end

function ∇relu(x::AbstractArray{T,N}, dy::AbstractArray{T,N}; nthreads::UInt64 = NNPACK_CPU_THREADS) where {T,N}
    T == Float32 || error("NNPACK RELU GRADIENT supports only Float32")
    N != 1 || error("NNPACK RELU GRADIENT requires an array of 2 or more dimensions")
    nnp_relu_input_gradient(x, dy, similar(x), threadpool = pthreadpool_create(nthreads))
end

function leakyrelu(x::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.01; inplace::Bool = true, nthreads::UInt64 = NNPACK_CPU_THREADS) where {T,N}
    T == Float32 || error("NNPACK RELU supports only Float32")
    N != 1 || error("NNPACK RELU requires an array of 2 or more dimensions")
    nnp_relu_output(x, inplace ? x : similar(x), negative_slope = negative_slope, threadpool = pthreadpool_create(nthreads))
end

function ∇leakyrelu(x::AbstractArray{T,N}, dy::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.0; nthreads::UInt64 = NNPACK_CPU_THREADS) where {T,N}
    T == Float32 || error("NNPACK LEAKY RELU GRADIENT supports only Float32")
    N != 1 || error("NNPACK LEAKY RELU GRADIENT requires an array of 2 or more dimensions")
    nnp_relu_input_gradient(x, dy, similar(x), negative_slope = negative_slope, threadpool = pthreadpool_create(nthreads))
end

function softmax(x::AbstractVecOrMat{T}; nthreads::UInt64 = NNPACK_CPU_THREADS) where T
    T == Float32 || error("NNPACK SOFTMAX supports only Float32")
    nnp_softmax_output(x, similar(x), threadpool = pthreadpool_create(nthreads))
end

function softmax!(x::AbstractVecOrMat{T}; inplace::Bool = true, nthreads::UInt64 = NNPACK_CPU_THREADS) where T
    T == Float32 || error("NNPACK SOFTMAX supports only Float32")
    nnp_softmax_output(x, inplace ? x : similar(x), threadpool = pthreadpool_create(nthreads))
end

function softmax!(y::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}; inplace::Bool = true, nthreads::UInt64 = NNPACK_CPU_THREADS) where T
    T == Float32 || error("NNPACK SOFTMAX supports only Float32")
    nnp_softmax_output(x, y, threadpool = pthreadpool_create(nthreads))
end

function maxpool(x::AbstractArray{T,4}, k::Tuple; pad = map(_->0,k), stride = k, nthreads::UInt64 = NNPACK_CPU_THREADS) where T
    T == Float32 || error("NNPACK MAXPOOL 2D supports only Float32")
    maxpool!(similar(x, pdims(size(x), k, expand(Val{length(k)}, pad), expand(Val{length(k)}, stride))), x, k, pad = pad, stride = stride, threadpool = pthreadpool_create(nthreads))
end

maxpool!(y::AbstractArray{Float32,4}, x::AbstractArray{Float32,4}, k::Tuple; pad = map(_->0,k), stride = k, threadpool = nothing) =
    nnp_max_pooling_output(x, y, k, padding = pad, stride = stride, threadpool = threadpool)

#NOTE: The API for profiling in not exposed. Also profiling is not functional currently
#NOTE: Mixed precision fully_connected inference and normal inference is not exposed

function fullyconnected(x::AbstractArray{T,2}, w::AbstractArray{T,2}; nthreads::UInt64 = NNPACK_CPU_THREADS) where T
    T == Float32 || error("NNPACK FULLY CONNECTED supports only Float32")
    nnp_fully_connected_output(x, w, zeros(Float32, size(w,1), size(x,2)), threadpool = pthreadpool_create(nthreads))
end

function conv(x::AbstractArray{T,4}, w::AbstractArray{T,4}; pad = 0, stride = 1, dilation = 1, algo = 0, nthreads::UInt64 = NNPACK_CPU_THREADS) where T
    T == Float32 || error("NNPACK FULLY CONNECTED supports only Float32")
    dilation == 1 || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    y = similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_))
    b = zeros(Float32, size(y, 3))
    conv!(y, x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo), threadpool = pthreadpool_create(nthreads))
end

function conv(x::AbstractArray{T,4}, w::AbstractArray{T,4}, b::AbstractArray{T,4}; pad = 0, stride = 1, dilation = 1, algo = 0, nthreads::UInt64 = NNPACK_CPU_THREADS) where T
    T == Float32 || error("NNPACK FULLY CONNECTED supports only Float32")
    dilation == 1 || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    conv!(similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_)), x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = algo, threadpool = pthreadpool_create(nthreads))
end

conv!(y::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}, b::AbstractArray{T,1}; pad = 0, stride = 1, dilation = 1, algo = 0, threadpool = nothing) where T =
    nnp_convolution_output(y, x, w, b, algo = algo, padding = pad, stride = stride, threadpool = threadpool)

function ∇conv_data(dy::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}; pad = 0, stride = 1, dilation = 1, algo = 0, threadpool = nothing) where T
    T == Float32 || error("NNPACK FULLY CONNECTED supports only Float32")
    dilation == 1 || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ∇conv_data!(zeros(Float32, size(x)), dy, x, w; pad = pad, stride = stride, dilation = dilation, algo = algo, threadpool = pthreadpool_create(nthreads))
end

∇conv_data!(dx::AbstractArray{T,4}, dy::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}; pad = 0, stride = 1, dilation = 1, algo = 0, threadpool = nothing) where T =
    nnp_convolution_input_gradient(dx, x, dy, w, padding = pad, stride = stride, dilation = dilation, algo = algo, threadpool = threadpool)

function ∇conv_filter(dy::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}; pad = 0, stride = 1, dilation = 1, algo = 0, threadpool = nothing) where T
    T == Float32 || error("NNPACK FULLY CONNECTED supports only Float32")
    dilation == 1 || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ∇conv_filter!(zeros(Float32, size(w)), dy, x, w; pad = pad, stride = stride, dilation = dilation, algo = algo, threadpool = pthreadpool_create(nthreads))
end

∇conv_filter!(dw::AbstractArray{T,4}, dy::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}; pad = 0, stride = 1, dilation = 1, algo = 0, threadpool = nothing) where T =
    nnp_convolution_kernel_gradient(dw, x, dy, w, padding = pad, stride = stride, dilation = dilation, algo = algo, threadpool = threadpool)
