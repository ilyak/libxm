# libxm 2.0 (beta)

Libxm is a distributed-parallel C/C++ library that provides routines for
efficient operations (e.g., contractions) on very large (terabytes in size)
disk-based block-tensors.

With libxm tensors can be stored on hard disks which allows for virtually
unlimited data size. Data are asynchronously prefetched to main memory for fast
access. Tensor contractions are reformulated as multiplications of matrices
done in batches using optimized BLAS routines. Tensor block-level symmetry and
sparsity is used to decrease storage and computational requirements. Libxm
supports single and double precision scalar and complex numbers.

### Reference

[I.A. Kaliman and A.I. Krylov, JCC 2017](https://dx.doi.org/10.1002/jcc.24713)

The code described in the paper can be found in the **xm1** branch.

### Compilation

To compile libxm you need a POSIX environment, an efficient BLAS library, and
an ANSI C complaint compiler. Issue `make` in the directory with libxm source
code to compile the library. To use libxm in your project, include `xm.h` file
and compile the code:

    cc -fopenmp myprog.c xm.a -lblas -lm

Detailed documentation can be found in `xm.h` and other header files. The tests
can be executed by issuing the following command in the directory with the
source code:

    make check

Compiler and flags can be adjusted by modifying libxm Makefile.
MPI support is enabled by defining `WITH_MPI` during compilation and
using `mpicc` as a compiler.

### Source code overview

- example.c - sample code with comments - start here
- xm.h - main libxm include header file
- tensor.c/tensor.h - block-tensor manipulation routines
- alloc.c/alloc.h - MPI-aware thread-safe disk-backed memory allocator
- blockspace.c/blockspace.h - operations on block-spaces
- dim.c/dim.h - operations on multidimensional indices
- test.c - testing facilities

Corresponding documentation can be found in individual header files.

### Parallel scaling

The table below shows parallel scalability of some libxm operations on the
NERSC Cori Cray XC40 supercomputer. The total tensor data size was over 2 Tb.
Burst Buffer was used in all tests. Table shows time in seconds with speedup
relative to 1 node shown in parenthesis.

|      Nodes      |  xm\_contract  |   xm\_add   |   xm\_set   |
|:---------------:|:--------------:|:-----------:|:-----------:|
|  1 (32 cores)   |  23660 (1.0x)  | 787 (1.0x)  | 457 (1.0x)  |
|  2 (64 cores)   |  11771 (2.0x)  | 436 (1.8x)  | 324 (1.4x)  |
|  4 (128 cores)  |   5938 (4.0x)  | 203 (3.9x)  | 115 (4.0x)  |
|  8 (256 cores)  |   3167 (7.5x)  | 168 (4.7x)  |  66 (6.9x)  |
| 16 (512 cores)  |   1606 (14.7x) |  69 (11.4x) |  28 (16.3x) |
| 32 (1024 cores) |    836 (28.3x) |  32 (24.6x) |  21 (21.8x) |

### Libxm users

- libxm is integrated with the [Q-Chem](http://www.q-chem.com) quantum
  chemistry package to accelerate large electronic structure calculations
- libxm is used as a backend in C++ tensor library libtensor
