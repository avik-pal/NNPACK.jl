# NNlib utility functions
using NNlib: expand, cdims, head, padtuple, psize,
             dilation_dims, pdims

# Functions to be overloaded
using NNlib: relu, leakyrelu, softmax!, softmax
             maxpool, maxpool!

export relu, leakyrelu, softmax!, softmax, maxpool,
       maxpool!

const AA{N} = AbstractArray{Float32,N}
const AA1 = Union{AA{2}, AA{3}, AA{4}, AA{5}}

function relu(x::AA1; inplace::Bool = true, nthreads::Int = 0)
    y = inplace ? x : similar(x)
    nnp_relu_output(x, y, threadpool = pthreadpool_create(nthreads))
end

function leakyrelu(x::AA1, a = oftype(x/1, 0.01); inplace::Bool = true, nthreads::Int = 0)
    y = inplace ? x : similar(x)
    nnp_relu_output(x, y, negative_slope = a, threadpool = pthreadpool_create(nthreads))
end

function softmax!(x::AbstractVecOrMat{Float32}; inplace::Bool = true, nthreads::Int = 0)
    y = inplace ? x : similar(x)
    nnp_softmax_output(x, y, threadpool = pthreadpool_create(nthreads))
end

softmax!(y::AbstractVecOrMat{Float32},x::AbstractVecOrMat{Float32}; inplace::Bool = true, nthreads::Int = 0) =
    nnp_softmax_output(x, y, threadpool = pthreadpool_create(nthreads))

function softmax(x::AbstractVecOrMat{Float32}; nthreads::Int = 0)
    y = similar(x)
    nnp_softmax_output(x, y, threadpool = pthreadpool_create(nthreads))
end

function maxpool(x::AA{2}, k; pad = map(_->0,k), stride = k, nthreads::Int = 0)
    y = similar(x, pdims(size(x), kernel, expand(Val{length(kernel)}, padding), expand(Val{length(kernel)}, stride)))
    maxpool!(x, y, k, padding = pad, stride = stride, threadpool = pthreadpool_create(nthreads))
end

maxpool!(y::AA{2}, x::AA{2}, k; pad = map(_->0,k), stride = k, threadpool = nothing) =
    nnp_max_pooling_output(x, y, k, padding = pad, stride = stride, threadpool = threadpool)
