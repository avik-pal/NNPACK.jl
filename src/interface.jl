export relu, leaky_relu, ∇relu, ∇leaky_relu,
       softmax, fully_connected, maxpool2d

function relu(x::AbstractArray{T,N}; inplace::Bool = true, nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU supports only Float32")
    N != 1 || error("NNPACK RELU requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_output(x, inplace = inplace, threadpool = threadpool)
end

function ∇relu(x::AbstractArray{T,N}, dy::AbstractArray{T,N}; nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU GRADIENT supports only Float32")
    N != 1 || error("NNPACK RELU GRADIENT requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_input_gradient(x, dy, threadpool = threadpool)
end

function leakyrelu(x::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.01; inplace::Bool = true, nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU supports only Float32")
    N != 1 || error("NNPACK RELU requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_output(x, inplace = inplace, negative_slope = negative_slope, threadpool = threadpool)
end

function ∇leakyrelu(x::AbstractArray{T,N}, dy::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.0; nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK LEAKY RELU GRADIENT supports only Float32")
    N != 1 || error("NNPACK LEAKY RELU GRADIENT requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_input_gradient(x, dy, negative_slope = negative_slope, threadpool = threadpool)
end

function softmax(x::AbstractVecOrMat{T}; inplace::Bool = true, nthreads::Int = 0) where T
    T == Float32 || error("NNPACK SOFTMAX supports only Float32")
    threadpool = pthreadpool_create(nthreads)
    nnp_softmax_output(x, inplace = inplace, threadpool = threadpool)
end

#NOTE: The API for profiling in not exposed. Also profiling is not functional currently
#NOTE: Mixed precision fully_connected inference and normal inference is not exposed

function fully_connected(x::AbstractArray{T,2}, w::AbstractArray{T,2}; nthreads::Int = 0) where T
    T == Float32 || error("NNPACK FULLY CONNECTED supports only Float32")
    threadpool = pthreadpool_create(nthreads)
    nnp_fully_connected_output(x, w, threadpool = threadpool)
end

function maxpool2d(x::AbstractArray{T,4}, kernel::Tuple; stride = 1, pad = 0, nthreads::Int = 0) where T
    T == Float32 || error("NNPACK MAXPOOL 2D supports only Float32")
    threadpool = pthreadpool_create(nthreads)
    y = similar(x, pdims(size(x), kernel, expand(Val{length(kernel)}, padding), expand(Val{length(kernel)}, stride)))
    nnp_max_pooling_output(x, y, kernel, padding = pad, stride = stride, threadpool = threadpool)
end
