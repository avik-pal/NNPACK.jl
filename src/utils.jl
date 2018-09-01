# Most of these functions have been borrowed from NNlib.jl

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
