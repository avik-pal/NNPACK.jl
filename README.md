# [WIP] NNPACK.jl

This is a wrapper for the low level deep learning acceleration package
for multicore CPUs [NNPACK](https://github.com/Maratyszcza/NNPACK).

### Usage Instructions

There are currently 2 API's for using this package. Presence of [NNlib](https://github.com/FluxML/NNlib.jl) allows direct integration with [Flux](https://github.com/FluxML/Flux.jl).

Currently the API is not documented but the exposed part of it is `interface.jl` and for NNlib part it's in `nnlib.jl`.

### Installation

This package works on julia 1.0. So to install it simply do

```julia
] add https://github.com/avik-pal/NNPACK.jl
```

This is currently a __Work In Progress__. But, feel free to open an issue if you encounter one.
