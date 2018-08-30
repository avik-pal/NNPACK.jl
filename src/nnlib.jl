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
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_output(x, inplace = inplace, threadpool = threadpool)
end

function leakyrelu(x::AA1, a = oftype(x/1, 0.01); inplace::Bool = true, nthreads::Int = 0)
    threadpool = pthreadpool_create(nthreads)
    nnp_relu_output(x, inplace = inplace, negative_slope = a, threadpool = threadpool)
end

function softmax!(x::AbstractVecOrMat{Float32}; nthreads::Int = 0)
    threadpool = pthreadpool_create(nthreads)
    nnp_softmax_output(x, inplace = true, threadpool = threadpool)
end

function softmax(x::AbstractVecOrMat{Float32}; nthreads::Int = 0)
    threadpool = pthreadpool_create(nthreads)
    nnp_softmax_output(x, inplace = false, threadpool = threadpool)
end

function maxpool(x::AA{2}, k; pad = map(_->0,k), stride = k, nthreads::Int = 0)
    threadpool = pthreadpool_create(nthreads)
    y = similar(x, pdims(size(x), kernel, expand(Val{length(kernel)}, padding), expand(Val{length(kernel)}, stride)))
    maxpool!(x, y, k, padding = pad, stride = stride, threadpool = threadpool)
    y
end

maxpool!(y::AA{2}, x::AA{2}, k; pad = map(_->0,k), stride = k, threadpool = pthreadpool_create()) =
    nnp_max_pooling_output(x, y, k, padding = pad, stride = stride, threadpool = threadpool)
