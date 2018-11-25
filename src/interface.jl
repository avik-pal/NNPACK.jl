softmax!(x::A) where A<:AbstractVecOrMat{Float32} =
    nnp_softmax_output(x, x, threadpool = shared_threadpool[])

softmax!(y::A, x::A) where A<:AbstractVecOrMat{Float32} =
    nnp_softmax_output(x, y, threadpool = shared_threadpool[])

softmax(x::A) where A<:AbstractVecOrMat{Float32} =
    nnp_softmax_output(x, similar(x), threadpool = shared_threadpool[])

function maxpool(x::A, k; pad = map(_->0,k), stride = k) where A<:AbstractArray{Float32, 4}
    pad_, stride_= check_support(x, k, pad, stride)
    maxpool!(similar(x, pdims(size(x), k, pad_, stride_)), x, k, pad = pad_, stride = stride_)
end

maxpool!(y::A, x::A, k; pad = map(_->0,k), stride = k) where A<:AbstractArray{Float32, 4} =
    nnp_max_pooling_output(x, y, k, padding = expand(Val{length(k)}, pad), stride = expand(Val{length(k)}, stride), threadpool = shared_threadpool[])

function conv(x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where A<:AbstractArray{Float32, 4}
    pad_, stride_ = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    y = similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_))
    conv!(y, x, w, zeros(Float32, size(y, 3)), pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo))
end

function conv(x::A1, w::A1, b::A2; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where {A1<:AbstractArray{Float32, 4}, A2<:AbstractArray{Float32, 1}}
    pad_, stride_ = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    y = similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_))
    conv!(y, x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo))
end

function conv!(y::A1, x::A1, w::A1, b::A2; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where {A1<:AbstractArray{Float32, 4}, A2<:AbstractArray{Float32, 1}}
    flipkernel == 0 && (w = reverse(reverse(w, dims=1), dims=2))
    nnp_convolution_output(y, x, w, b, algo = algo, padding = pad, stride = stride, threadpool = shared_threadpool[])
end

function ∇conv_data(dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where A<:AbstractArray{Float32, 4}
    pad_, stride_ = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    ∇conv_data!(zeros(Float32, size(x)), dy, x, w; pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo))
end

function ∇conv_data!(dx::A, dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A<:AbstractArray{Float32, 4}
    flipkernel == 0 && (w = reverse(reverse(w, dims=1), dims=2))
    nnp_convolution_input_gradient(dx, x, dy, w, padding = pad, stride = stride, algo = algo, threadpool = shared_threadpool[])
end

function ∇conv_filter(dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where A<:AbstractArray{Float32, 4}
    pad_, stride_ = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    ∇conv_filter!(zeros(Float32, size(w)), dy, x, w; pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo))
end

function ∇conv_filter!(dw::A, dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A<:AbstractArray{Float32, 4}
    flipkernel == 0 && (w = reverse(reverse(w, dims=1), dims=2))
    dw .= nnp_convolution_kernel_gradient(dw, x, dy, w, padding = pad, stride = stride, algo = algo, threadpool = shared_threadpool[])
    flipkernel == 0 ? reverse(reverse(dw, dims=1), dims=2) : dw
end
