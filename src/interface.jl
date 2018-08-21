function relu(x::AbstractArray{T,N}; inplace::Bool = true, nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU supports only Float32")
    N != 1 || error("NNPACK RELU requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_output(x, inplace = inplace, threadpool = threadpool)
end

function leaky_relu(x::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.01; inplace::Bool = true, nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU supports only Float32")
    N != 1 || error("NNPACK RELU requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_output(x, inplace = inplace, negative_slope = negative_slope, threadpool = threadpool)
end

function ∇relu(x::AbstractArray{T,N}, Δy::AbstractArray{T,N}; inplace::Bool = true, nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU GRADIENT supports only Float32")
    N != 1 || error("NNPACK RELU GRADIENT requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_input_gradient(x, Δy, inplace = inplace, threadpool = threadpool)
end

function ∇leaky_relu(x::AbstractArray{T,N}, Δy::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.0; inplace::Bool = true, nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK LEAKY RELU GRADIENT supports only Float32")
    N != 1 || error("NNPACK LEAKY RELU GRADIENT requires an array of 2 or more dimensions")
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_input_gradient(x, Δy, inplace = inplace, negative_slope = negative_slope, threadpool = threadpool)
end
