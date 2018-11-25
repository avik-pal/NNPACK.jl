function softmax!(x::A) where A<:AbstractVecOrMat
    x = Float32.(x)
    softmax!(x)
end

softmax!(y::A, x::B) where {A<:AbstractVecOrMat, B<:AbstractVecOrMat} = softmax!(Float32.(y), Float32.(x))

softmax(x::A) where A<:AbstractVecOrMat = softmax(Float32.(x))

maxpool(x::A, k; pad = map(_->0,k), stride = k) where A<:AbstractArray{T, 4} where T =
    maxpool(Float32.(x), k, pad = pad, stride = stride)

maxpool!(y::A, x::B, k; pad = map(_->0,k), stride = k) where {A<:AbstractArray{T, 4} where T, B<:AbstractArray{T, 4} where T} =
    maxpool!(Float32.(y), Float32.(x), k, pad = pad, stride = stride)

conv(x::A, w::B; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where {A<:AbstractArray{T, 4} where T, B<:AbstractArray{T, 4} where T} =
    conv(Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo)

conv(x::A, w::B, b::C; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where {A<:AbstractArray{T, 4} where T, B<:AbstractArray{T, 4} where T, C<:AbstractArray{T, 1} where T} =
    conv(Float32.(x), Float32.(w), Float32.(b), pad = pad, stride = stride, dilation = dilation, algo = algo)

conv!(y::A, x::B, w::C, b::D; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where {A<:AbstractArray{T, 4} where T, B<:AbstractArray{T, 4} where T, C<:AbstractArray{T, 4} where T, D<:AbstractArray{T, 1} where T} =
    conv!(Float32.(y), Float32.(x), Float32.(w), Float32.(b), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel)

∇conv_data(dy::A, x::B, w::C; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where {A<:AbstractArray{T, 4} where T, B<:AbstractArray{T, 4} where T, C<:AbstractArray{T, 4} where T} =
    ∇conv_data(Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo)

∇conv_data!(dx::A, dy::B, x::C, w::D; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where {A<:AbstractArray{T, 4} where T, B<:AbstractArray{T, 4} where T, C<:AbstractArray{T, 4} where T, D<:AbstractArray{T, 4} where T} =
    ∇conv_data!(Float32.(dx), Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel)

∇conv_filter(dy::A, x::B, w::C; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where {A<:AbstractArray{T, 4} where T, B<:AbstractArray{T, 4} where T, C<:AbstractArray{T, 4} where T} =
    ∇conv_filter(Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo)

∇conv_filter!(dw::A, dy::B, x::C, w::D; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where {A<:AbstractArray{T, 4} where T, B<:AbstractArray{T, 4} where T, C<:AbstractArray{T, 4} where T, D<:AbstractArray{T, 1} where T} =
    ∇conv_filter!(Float32.(dw), Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel)
