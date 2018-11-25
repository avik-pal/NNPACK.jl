# NNPACK.jl

This package has been shifted to __NNlib.jl__ and any furthur support will be provided there. The package will continue to exist here if someone wants to use it without Flux.

This is a wrapper for the low level deep learning acceleration package
for multicore CPUs [NNPACK](https://github.com/Maratyszcza/NNPACK).

### Usage Instructions

There are currently 2 API's for using this package. Presence of [NNlib](https://github.com/FluxML/NNlib.jl) allows direct integration with [Flux](https://github.com/FluxML/Flux.jl).

Currently the API is not documented but the exposed part of it is `interface.jl` and for NNlib part it's in `nnlib.jl`.

To change the `NNPACK_NUM_THREADS` change the ENVIRONMENT VARIABLE `JULIA_NUM_THREADS` and load NNPACK. Ideally keep the `NNPACK_NUM_THREADS` between __4__ and __8__.

### Installation

This package works on julia 1.0. So to install it simply do

```julia
] add https://github.com/avik-pal/NNPACK.jl
```

This will work only on Linux. Follow the below instructions to get it working on __OSX__

```
# Install PeachPy
$ git clone https://github.com/Maratyszcza/PeachPy.git
$ cd PeachPy
$ sudo pip install --upgrade -r requirements.txt
$ python setup.py generate
$ sudo pip install --upgrade .

# Install Ninja Build System
$ sudo apt-get install ninja-build
$ pip install ninja-syntax

# Build NNPack shared library
$ cd ~
$ git clone --recursive https://github.com/Maratyszcza/NNPACK.git
$ cd NNPACK
$ python ./configure.py
$ cmake . -G Ninja -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DNNPACK_LIBRARY_TYPE="shared"
```

After this copy the `libnnpack.so` to `~/.julia/packages/NNPACK/deps/usr/lib/libnnpack.so`.

### Known Issues

1. __Windows__ is not supported by NNPACK. There is unofficial support for windows in [nnpack-windows](https://github.com/zeno40/nnpack-windows) but the API is a bit different.
2. __OSX__ build will fail.
3. __Travis tests__ fail due to Unsupported Hardware error.
2. `BenchmarkTools` fails to time the code if `NNPACK_NUM_THREADS` is not set to `0`.
