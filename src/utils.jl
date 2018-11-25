# Code borrowed from NNlib

expand(::Type{Val{N}}, i::Integer) where N = ntuple(_ -> i, Val(N))
expand(::Type{Val{N}}, i::NTuple{N, Integer}) where N = i

function cdims(x::NTuple{N}, w::NTuple{N}, pad, stride) where N
    ntuple(Val{N}) do i
        if i < N-1
            1 + div(x[i] - w[i] + 2*pad[i], stride[i])
        elseif i == N-1
            w[N]
        else # i == N
            x[N]
        end
    end
end

head(x) = reverse(Base.tail(reverse(x)))
padtuple(x::Tuple,p::Integer) = map(_->p, head(head(x)))
padtuple(x::Tuple,p::Tuple) = p
padtuple(x::AbstractArray,p) = padtuple(size(x),p)

function psize(p, x)
    nd = ndims(x)-2
    if isa(p,Number)
        fill(Int(p),nd)
    elseif length(p)==nd
        collect(Int,p)
    else
        throw(DimensionMismatch("psize: $p $nd"))
    end
end

function dilation_dims(w, dilation = 1)
    N = ndims(w)
    dims_w = size(w)
    dil = psize(dilation, w)
    ntuple(N) do i
        if i < N - 1
            (dims_w[i] - 1) * dil[i] + 1
        else
            dims_w[i]
        end
    end
end

function pdims(dims::Dims{N}, window, padding, stride) where N
    ntuple(Val(N)) do i
        if i < N-1
            1 + (dims[i] + 2*padding[i] - window[i])÷stride[i]
        else
            dims[i]
        end
    end
end

function check_support(x, k, pad, stride, dilation = 1)
    supported = true
    dilation == 1 || dilation == (1, 1) || (supported = false)
    pad_, stride_ = expand(Val{length(k)}, pad), expand(Val{length(k)}, stride)
    ((size(x, 1) - k[1] + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - k[2] + 2 * pad_[2]) % stride_[2] == 0) || (fallback = true)
    !supported && error("Operation Not Supported by NNPACK")
    return pad_, stride_
end
