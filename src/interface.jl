export relu, leakyrelu, ∇relu, ∇leakyrelu,
       softmax, fullyconnected, maxpool

function relu(x::AbstractArray{T,N}; inplace::Bool = true, nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU supports only Float32")
    N != 1 || error("NNPACK RELU requires an array of 2 or more dimensions")
    y = inplace ? x : similar(x)
    nnp_relu_output(x, y, threadpool = pthreadpool_create(nthreads))
end

function ∇relu(x::AbstractArray{T,N}, dy::AbstractArray{T,N}; nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU GRADIENT supports only Float32")
    N != 1 || error("NNPACK RELU GRADIENT requires an array of 2 or more dimensions")
    nnp_relu_input_gradient(x, dy, similar(x), threadpool = pthreadpool_create(nthreads))
end

function leakyrelu(x::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.01; inplace::Bool = true, nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK RELU supports only Float32")
    N != 1 || error("NNPACK RELU requires an array of 2 or more dimensions")
    y = inplace ? x : similar(x)
    nnp_relu_output(x, y, negative_slope = negative_slope, threadpool = pthreadpool_create(nthreads))
end

function ∇leakyrelu(x::AbstractArray{T,N}, dy::AbstractArray{T,N}, negative_slope::AbstractFloat = 0.0; nthreads::Int = 0) where {T,N}
    T == Float32 || error("NNPACK LEAKY RELU GRADIENT supports only Float32")
    N != 1 || error("NNPACK LEAKY RELU GRADIENT requires an array of 2 or more dimensions")
    nnp_relu_input_gradient(x, dy, similar(x), negative_slope = negative_slope, threadpool = pthreadpool_create(nthreads))
end

function softmax(x::AbstractVecOrMat{T}; nthreads::Int = 0) where T
    T == Float32 || error("NNPACK SOFTMAX supports only Float32")
    y = similar(x)
    nnp_softmax_output(x, y, threadpool = pthreadpool_create(nthreads))
end

function softmax!(x::AbstractVecOrMat{T}; inplace::Bool = true, nthreads::Int = 0) where T
    T == Float32 || error("NNPACK SOFTMAX supports only Float32")
    y = inplace ? x : similar(x)
    nnp_softmax_output(x, y, threadpool = pthreadpool_create(nthreads))
end

function softmax!(y::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}; inplace::Bool = true, nthreads::Int = 0) where T
    T == Float32 || error("NNPACK SOFTMAX supports only Float32")
    nnp_softmax_output(x, y, threadpool = pthreadpool_create(nthreads))
end

function maxpool(x::AbstractArray{T,4}, k::Tuple; pad = map(_->0,k), stride = k, nthreads::Int = 0) where T
    T == Float32 || error("NNPACK MAXPOOL 2D supports only Float32")
    y = similar(x, pdims(size(x), k, expand(Val{length(k)}, padding), expand(Val{length(k)}, stride)))
    maxpool!(x, y, k, padding = pad, stride = stride, threadpool = pthreadpool_create(nthreads))
end

maxpool!(y::AbstractArray{Float32,4}, x::AbstractArray{Float32,4}, k::Tuple; pad = map(_->0,k), stride = k, threadpool = nothing) =
    nnp_max_pooling_output(x, y, k, padding = pad, stride = stride, threadpool = threadpool)

#NOTE: The API for profiling in not exposed. Also profiling is not functional currently
#NOTE: Mixed precision fully_connected inference and normal inference is not exposed

function fullyconnected(x::AbstractArray{T,2}, w::AbstractArray{T,2}; nthreads::Int = 0) where T
    T == Float32 || error("NNPACK FULLY CONNECTED supports only Float32")
    y = zeros(Float32, size(w,1), size(x,2))
    nnp_fully_connected_output(x, w, y, threadpool = pthreadpool_create(nthreads))
end
