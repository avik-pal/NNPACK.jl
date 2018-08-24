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

function leaky_relu(x::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.01; inplace::Bool = true, nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU supports only Float32")
    N != 1 || error("NNPACK RELU requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_output(x, inplace = inplace, negative_slope = negative_slope, threadpool = threadpool)
end

function ∇leaky_relu(x::AbstractArray{T,N}, dy::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.0; nthreads::Int = 0) where {T,N}
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
