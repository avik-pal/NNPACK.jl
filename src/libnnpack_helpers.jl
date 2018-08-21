const nnp_status = UInt32
const nnp_status_success = (UInt32)(0)
const nnp_status_invalid_batch_size = (UInt32)(2)
const nnp_status_invalid_channels = (UInt32)(3)
const nnp_status_invalid_input_channels = (UInt32)(4)
const nnp_status_invalid_output_channels = (UInt32)(5)
const nnp_status_invalid_input_size = (UInt32)(10)
const nnp_status_invalid_input_stride = (UInt32)(11)
const nnp_status_invalid_input_padding = (UInt32)(12)
const nnp_status_invalid_kernel_size = (UInt32)(13)
const nnp_status_invalid_pooling_size = (UInt32)(14)
const nnp_status_invalid_pooling_stride = (UInt32)(15)
const nnp_status_invalid_algorithm = (UInt32)(16)
const nnp_status_invalid_transform_strategy = (UInt32)(17)
const nnp_status_invalid_output_subsampling = (UInt32)(13)
const nnp_status_invalid_activation = (UInt32)(14)
const nnp_status_invalid_activation_parameters = (UInt32)(15)
const nnp_status_unsupported_input_size = (UInt32)(20)
const nnp_status_unsupported_input_stride = (UInt32)(21)
const nnp_status_unsupported_input_padding = (UInt32)(22)
const nnp_status_unsupported_kernel_size = (UInt32)(23)
const nnp_status_unsupported_pooling_size = (UInt32)(24)
const nnp_status_unsupported_pooling_stride = (UInt32)(25)
const nnp_status_unsupported_algorithm = (UInt32)(26)
const nnp_status_unsupported_transform_strategy = (UInt32)(57)
const nnp_status_unsupported_activation = (UInt32)(28)
const nnp_status_unsupported_activation_parameters = (UInt32)(29)
const nnp_status_uninitialized = (UInt32)(50)
const nnp_status_unsupported_hardware = (UInt32)(51)
const nnp_status_out_of_memory = (UInt32)(52)
const nnp_status_insufficient_buffer = (UInt32)(53)
const nnp_status_misaligned_buffer = (UInt32)(54)

const nnp_activation = UInt32
const nnp_activation_identity = (UInt32)(0)
const nnp_activation_relu = (UInt32)(1)

const nnp_convolution_algorithm = UInt32
const nnp_convolution_algorithm_auto = (UInt32)(0)
const nnp_convolution_algorithm_ft8x8 = (UInt32)(1)
const nnp_convolution_algorithm_ft16x16 = (UInt32)(2)
const nnp_convolution_algorithm_wt8x8 = (UInt32)(3)
const nnp_convolution_algorithm_implicit_gemm = (UInt32)(4)
const nnp_convolution_algorithm_direct = (UInt32)(5)
const nnp_convolution_algorithm_wt8x8_fp16 = (UInt32)(6)

const nnp_convolution_transform_strategy = UInt32
const nnp_convolution_transform_strategy_compute = (UInt32)(1)
const nnp_convolution_transform_strategy_precompute = (UInt32)(2)
const nnp_convolution_transform_strategy_reuse = (UInt32)(3)

const pthreadpool_t = Ptr{Nothing}

struct nnp_size
	width::Csize_t
	height::Csize_t
end

struct nnp_padding
	top::Csize_t
	right::Csize_t
	bottom::Csize_t
	left::Csize_t
end

struct nnp_profile
	total::Cdouble
	input_transform::Cdouble
	kernel_transform::Cdouble
	output_transform::Cdouble
	block_multiplication::Cdouble
end
