# libxm 2.0 (beta)

Libxm is a library that provides routines for efficient contractions of very
large (terabytes in size) disk-based block-tensors on multi-core CPUs, GPUs,
and various floating point accelerators.

With libxm tensors can be stored on hard disks which allow virtually unlimited
data size. Data are asynchronously prefetched to main memory for fast access.
Tensor contractions are reformulated as multiplications of big matrices done in
batches. Tensor symmetry and sparsity is used to decrease storage and
computational requirements. Computations can be efficiently accelerated using
multiple GPUs or other accelerators like Intel Xeon Phi. For very large
problems libxm shows considerable speedups compared to similar tensor
contraction codes. Libxm supports single and double precision scalar and
complex numbers.

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

    cc myprog.c xm.c alloc.c blockspace.c -lblas -lpthread -lm

Replace `-lblas` with appropriate accelerated libraries (e.g. `-lnvblas`) to
get the benefits of corresponding hardware. Detailed documentation can be
found in `xm.h` file. The tests can be run by issuing the following commands
in the directory with the source code.

    make
    make check

Compiler and flags can be adjusted by modifying the Makefile.

### Source code overview

- xm.h - public API header with documentation
- xm.c - libxm implementation code
- alloc.c/alloc.h - disk-backed allocator for large tensors
- gcalloc.c/alloc.h - experimental garbage-collecting disk-backed allocator
- auxil.c/auxil.h - optional auxiliary functions for tensor creation
- test.c - facilities for randomized testing

### Users of libxm

- libxm is integrated with Q-Chem package to accelerate high-level quantum
  chemistry calculations
- libxm is used by a C++ tensor library libtensor
