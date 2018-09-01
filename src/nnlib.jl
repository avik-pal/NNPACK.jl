# NNlib utility functions
using NNlib: expand, cdims, head, padtuple, psize,
             dilation_dims, pdims

# Functions to be overloaded
import NNlib: relu, leakyrelu, softmax!, softmax, maxpool, maxpool!,
              conv, ∇conv_data, ∇conv_filter, conv!, ∇conv_data!,
              ∇conv_filter!

# Overloaded function exports
export relu, leakyrelu, softmax, maxpool, conv, ∇conv_data,
       ∇conv_filter

const AA{N} = AbstractArray{Float32,N}
const AA1 = Union{AA{2}, AA{3}, AA{4}, AA{5}}

relu(x::AA1; inplace::Bool = true, nthreads::Int = 0) =
    nnp_relu_output(x, inplace ? x : similar(x), threadpool = pthreadpool_create(nthreads))

leakyrelu(x::AA1, a = oftype(x/1, 0.01); inplace::Bool = true, nthreads::Int = 0) =
    nnp_relu_output(x, inplace ? x : similar(x), negative_slope = a, threadpool = pthreadpool_create(nthreads))

softmax!(x::AbstractVecOrMat{Float32}; inplace::Bool = true, nthreads::Int = 0) =
    nnp_softmax_output(x, inplace ? x : similar(x), threadpool = pthreadpool_create(nthreads))

softmax!(y::AbstractVecOrMat{Float32},x::AbstractVecOrMat{Float32}; inplace::Bool = true, nthreads::Int = 0) =
    nnp_softmax_output(x, y, threadpool = pthreadpool_create(nthreads))

softmax(x::AbstractVecOrMat{Float32}; nthreads::Int = 0) =
    nnp_softmax_output(x, similar(x), threadpool = pthreadpool_create(nthreads))

maxpool(x::AA{4}, k; pad = map(_->0,k), stride = k, nthreads::Int = 0) =
    maxpool!(similar(x, pdims(size(x), k, expand(Val{length(k)}, pad), expand(Val{length(k)}, stride))), x, k, pad = expand(Val{length(k)}, pad), stride = expand(Val{length(k)}, stride), threadpool = pthreadpool_create(nthreads))

maxpool!(y::AA{4}, x::AA{4}, k; pad = map(_->0,k), stride = k, threadpool = nothing) =
    nnp_max_pooling_output(x, y, k, padding = expand(Val{length(k)}, pad), stride = expand(Val{length(k)}, stride), threadpool = threadpool)

function conv(x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = 0, nthreads::Int = 0)
    dilation == 1 || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    y = similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_))
    b = zeros(Float32, size(y, 3))
    conv!(y, x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = algo, threadpool = pthreadpool_create(nthreads))
end

function conv(x::AA{4}, w::AA{4}, b::AA{4}; pad = 0, stride = 1, dilation = 1, algo = 0, nthreads::Int = 0)
    dilation == 1 || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    conv!(similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_)), x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = algo, threadpool = pthreadpool_create(nthreads))
end

conv!(y::AA{4}, x::AA{4}, w::AA{4}, b::AA{1}; pad = 0, stride = 1, dilation = 1, algo = 0, threadpool = nothing) =
    nnp_convolution_output(y, x, w, b, algo = algo, padding = pad, stride = stride, threadpool = threadpool)
