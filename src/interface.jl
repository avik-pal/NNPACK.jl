function relu(x::AbstractArray{T,N}; inplace::Bool = false, nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU supports only Float32")
    N != 1 || error("NNPACK RELU requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_output(x, inplace = inplace, threadpool = threadpool)
end

relu!(x::AbstractArray{T,N}; nthreads::Int = 0) where {T,N} =
    relu(x, inplace = true, nthreads = nthreads)

function ∇relu(x::AbstractArray{T,N}, dy::AbstractArray{T,N}; nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU GRADIENT supports only Float32")
    N != 1 || error("NNPACK RELU GRADIENT requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_input_gradient(x, dy, threadpool = threadpool)
end

function leaky_relu(x::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.01; inplace::Bool = false, nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU supports only Float32")
    N != 1 || error("NNPACK RELU requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_output(x, inplace = inplace, negative_slope = negative_slope, threadpool = threadpool)
end

leaky_relu!(x::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.01; nthreads::Int = 0) where {T,N} =
    leaky_relu(x, negative_slope = negative_slope, inplace = true, nthreads = nthreads)

function ∇leaky_relu(x::AbstractArray{T,N}, dy::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.0; nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK LEAKY RELU GRADIENT supports only Float32")
    N != 1 || error("NNPACK LEAKY RELU GRADIENT requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_input_gradient(x, dy, negative_slope = negative_slope, threadpool = threadpool)
end

function softmax(x::AbstractVecOrMat{T}; inplace::Bool = false, nthreads::Int = 0) where T
    T == Float32 || error("NNPACK SOFTMAX supports only Float32")
    threadpool = pthreadpool_create(nthreads)
    nnp_softmax_output(x, inplace = inplace, threadpool = threadpool)
end

softmax!(x::AbstractVecOrMat{T}; nthreads::Int = 0) where {T} =
    softmax(x, inplace = true, nthreads = nthreads)
