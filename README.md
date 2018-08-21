# [WIP] NNPACK.jl

This is a wrapper for the low level deep learning acceleration package
for multicore CPUs [NNPACK](https://github.com/Maratyszcza/NNPACK).

To use this package currently we need a `libnnpack.so` file placed
inside the `src` folder. To generate the `libnnpack.so` go to [this link](https://github.com/Maratyszcza/NNPACK/issues/144) .This is pretty inefficient but a version with
pre-built binaries using BinaryBuilder is in works.


This is currently a __Work In Progress__.
