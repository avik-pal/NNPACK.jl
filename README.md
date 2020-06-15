# NOTE

This package is not officially maintained. It has been merged in [NNlib.jl](https://github.com/FluxML/NNlib.jl)
which offers deep learning primitives for a variety of backends. The source code for NNPACK.jl is now maintained
[here](https://github.com/FluxML/NNlib.jl/tree/master/src/nnpack) so please redirect any issue and pull requests
to NNlib.jl.

# NNPACK.jl

This is a wrapper for the low level deep learning acceleration package
for multicore CPUs [NNPACK](https://github.com/Maratyszcza/NNPACK).

### Usage Instructions

__NNPACK__ can be directly used with NNlib without any dependency on this package. The NNPACK wrapper
resides inside the NNlib package. For integration with other deep learning libraries this wrapper may
be used.

### Installation

This package works on julia 1.0. So to install it simply do

```julia
] add https://github.com/avik-pal/NNPACK.jl
```

This will work only on Linux. Follow the `Installation for OSX` to get it working on __OSX__

### Exported Functions

1. `conv`
2. `∇conv_data`
3. `∇conv_filter`
4. `softmax`
5. `maxpool`

To know more in detail about this functions refer to the `src/interface.jl` file. If you are familiar with the NNlib APIs then its pretty simple to understand this.

There are other NNPACK functions available but their API is not exposed. For using them refer to
the `src/libnnpack.jl` file.

### Environment Variables

1. `NNPACK_CPU_THREADS`: Controls the number of threads NNPACK is allowed to use. Defaults to `4`.
2. `NNPACK_FAST_OPS`: Enables operations on `Float64` Arrays. However, you might loose precision and the final output will also be in `Float32`. Defaults to `1`.

### Installation for OSX

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
