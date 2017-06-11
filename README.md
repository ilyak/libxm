# libxm 2.0 (beta)

Libxm is a C/C++ library that provides routines for efficient operations (e.g.,
contractions) on very large (terabytes in size) disk-based block-tensors on
multi-core CPUs, GPUs, and various floating point accelerators.

With libxm tensors can be stored on hard disks which allow virtually unlimited
data size. Data are asynchronously prefetched to main memory for fast access.
Tensor contractions are reformulated as multiplications of big matrices done in
batches. Tensor block-level symmetry and sparsity is used to decrease storage
and computational requirements. Computations can be accelerated using multiple
GPUs or other accelerators like Intel Xeon Phi. For very large problems libxm
shows considerable speedups compared to similar tensor contraction codes. Libxm
supports single and double precision scalar and complex numbers.

### Reference

[I.A. Kaliman and A.I. Krylov, JCC 2017](https://dx.doi.org/10.1002/jcc.24713)

The code described in the paper can be found in the **xm1** branch.

### Usage

A documented example on how to use libxm can be found in the `example.c` file.
Once tensors are setup the contraction routine is similar to BLAS dgemm call:

    xm_contract(alpha, A, B, beta, C, "abcd", "ijcd", "ijab");

This will preform the following contraction of two 4-index tensors A and B:

    C_ijab := alpha * A_abcd * B_ijcd + beta * C_ijab

### Compilation

To compile libxm you need a POSIX environment, an efficient BLAS library, and
an ANSI C complaint compiler. To use libxm in your project, include `xm.h` file
and compile the code:

    cc myprog.c alloc.c blockspace.c dim.c xm.c -lblas -lpthread -lm

Replace `-lblas` with appropriate accelerated libraries (e.g. `-lnvblas`) to
get the benefits of corresponding hardware. Detailed documentation can be
found in `xm.h` file. The tests can be run by issuing the following commands
in the directory with the source code.

    make
    make check

Compiler and flags can be adjusted by modifying the Makefile.

### Source code overview

- xm.h - public API header with documentation
- example.c - sample code with comments
- alloc.c/alloc.h - disk-backed allocator for large tensors
- blockspace.c/blockspace.h - operations on block-spaces
- dim.c/dim.h - operations on multidimensional indices
- tensor.c/tensor.h - block-tensor types
- test.c - testing facilities

### Libxm users

- libxm is integrated with the [Q-Chem](http://www.q-chem.com) quantum
  chemistry package to accelerate large electronic structure calculations
- libxm is used as a backend in C++ tensor library libtensor
