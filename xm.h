/*
 * Copyright (c) 2014-2017 Ilya Kaliman
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef XM_H_INCLUDED
#define XM_H_INCLUDED

#include "alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of tensor dimensions. */
#define XM_MAX_DIM 8

/* Block type. */
#define XM_BLOCK_TYPE_ZERO        0  /* see xm_tensor_set_zero_block */
#define XM_BLOCK_TYPE_CANONICAL   1  /* see xm_tensor_set_canonical_block */
#define XM_BLOCK_TYPE_DERIVATIVE  2  /* see xm_tensor_set_derivative_block */

#if defined(XM_SCALAR_FLOAT)
typedef float xm_scalar_t;
#elif defined(XM_SCALAR_DOUBLE_COMPLEX)
#include <complex.h>
typedef double complex xm_scalar_t;
#elif defined(XM_SCALAR_FLOAT_COMPLEX)
#include <complex.h>
typedef float complex xm_scalar_t;
#else /* assume double */
typedef double xm_scalar_t;
#endif

/* Opaque tensor structure. */
typedef struct xm_tensor xm_tensor_t;

/* Multidimensional block-space. */
typedef struct xm_block_space xm_block_space_t;

/* Multidimensional index. */
typedef struct {
	size_t n, i[XM_MAX_DIM];
} xm_dim_t;

/* Print libxm banner to the standard output. */
void xm_print_banner(void);


/* Operations on multidimensional indices. */

/* Initialize all indices of a dim to zero. */
xm_dim_t xm_dim_zero(size_t n);

/* Initialize all indices of a dim to the same value. */
xm_dim_t xm_dim_same(size_t n, size_t dim);

/* Initialize a 1-D dim. */
xm_dim_t xm_dim_1(size_t dim1);

/* Initialize a 2-D dim. */
xm_dim_t xm_dim_2(size_t dim1, size_t dim2);

/* Initialize a 3-D dim. */
xm_dim_t xm_dim_3(size_t dim1, size_t dim2, size_t dim3);

/* Initialize a 4-D dim. */
xm_dim_t xm_dim_4(size_t dim1, size_t dim2, size_t dim3, size_t dim4);

/* Return an n-dimensional identity permutation. */
xm_dim_t xm_dim_identity_permutation(size_t n);

/* Scale all dimensions of a dim by s. */
xm_dim_t xm_dim_scale(const xm_dim_t *dim, size_t s);

/* Return dot product of all indices of a dim. */
size_t xm_dim_dot(const xm_dim_t *dim);

/* Return non-zero if two dims are equal. */
int xm_dim_eq(const xm_dim_t *a, const xm_dim_t *b);

/* Return non-zero if two dims are not equal. */
int xm_dim_ne(const xm_dim_t *a, const xm_dim_t *b);

/* Return non-zero if an index is within zero and dim. */
int xm_dim_less(const xm_dim_t *idx, const xm_dim_t *dim);

/* Increment an index by one wrapping on dimensions. */
void xm_dim_inc(xm_dim_t *idx, const xm_dim_t *dim);

void xm_dim_set_mask(xm_dim_t *a, const xm_dim_t *ma, const xm_dim_t *b,
    const xm_dim_t *mb);

size_t xm_dim_dot_mask(const xm_dim_t *dim, const xm_dim_t *mask);

void xm_dim_inc_mask(xm_dim_t *idx, const xm_dim_t *dim, const xm_dim_t *mask);


/* Operations on tensors. */

/* Create a block-tensor. */
xm_tensor_t *xm_tensor_create(const xm_block_space_t *bs,
    xm_allocator_t *allocator);

/* Return a block-space associated with this tensor. */
const xm_block_space_t *xm_tensor_get_block_space(const xm_tensor_t *tensor);

/* Return an allocator associated with this tensor. */
xm_allocator_t *xm_tensor_get_allocator(xm_tensor_t *tensor);

/* Clone a tensor using given allocator. If allocator argument is NULL, the one
 * from tensor is used. */
xm_tensor_t *xm_tensor_clone(const xm_tensor_t *tensor,
    xm_allocator_t *allocator);

/* Return absolute tensor dimensions in total number of elements. */
xm_dim_t xm_tensor_get_abs_dims(const xm_tensor_t *tensor);

/* Return tensor dimensions in number of blocks. */
xm_dim_t xm_tensor_get_nblocks(const xm_tensor_t *tensor);

/* Return an individual element of a tensor given its absolute index.
 * Note: this function is relatively slow. */
xm_scalar_t xm_tensor_get_element(const xm_tensor_t *tensor,
    const xm_dim_t *idx);

/* Get elements of a particular block. The storage pointed to by data should
 * be large enough to store all elements of a block. */
void xm_tensor_get_block_elements(xm_tensor_t *tensor, const xm_dim_t *blkidx,
    xm_scalar_t *data, size_t data_bytes);

/* Return type of a block. This returns one of the XM_BLOCK_TYPE_ values. */
int xm_tensor_get_block_type(const xm_tensor_t *tensor, const xm_dim_t *blkidx);

/* Return dimensions of a specific block. */
xm_dim_t xm_tensor_get_block_dims(const xm_tensor_t *tensor,
    const xm_dim_t *blkidx);

/* Setup a zero-block (all elements of a block are zeros).
 * No actual data are stored. */
void xm_tensor_set_zero_block(xm_tensor_t *tensor, const xm_dim_t *blkidx);

/* Setup a canonical tensor block. Canonical blocks are the only ones that
 * store actual data.
 * Note: if blocks are allocated using a disk-backed allocator they should be
 * at least several megabytes in size for best performance (e.g., 32^4 elements
 * for 4-index tensors).
 * The data_ptr argument must be allocated using the same allocator that was
 * used during tensor creation. Allocation must be large enough to hold block
 * data. */
void xm_tensor_set_canonical_block(xm_tensor_t *tensor, const xm_dim_t *blkidx,
    uintptr_t data_ptr);

/* Setup a derivative block. A derivative block is a copy of some canonical
 * block with applied permutation and multiplication by a scalar factor.
 * No actual data are stored for derivative blocks. */
void xm_tensor_set_derivative_block(xm_tensor_t *tensor, const xm_dim_t *blkidx,
    const xm_dim_t *source_blkidx, const xm_dim_t *permutation,
    xm_scalar_t scalar);

/* Allocate storage sufficient to hold data for a particular block using
 * associated allocator. */
uintptr_t xm_tensor_allocate_block_data(xm_tensor_t *tensor,
    const xm_dim_t *blkidx);

/* Return block data pointer. This will return XM_NULL_PTR for zero and
 * derivative blocks. */
uintptr_t xm_tensor_get_block_data_ptr(xm_tensor_t *tensor,
    const xm_dim_t *blkidx);

/* Deallocate all block data associated with this tensor. */
void xm_tensor_free_block_data(xm_tensor_t *tensor);

/* Release resources associated with a tensor. The actual block data are not
 * freed by this function. */
void xm_tensor_free(xm_tensor_t *tensor);

/* Contract two tensors (c = alpha * a * b + beta * c) over contraction indices
 * specified by strings idxa and idxb. Permutation of tensor c is specified by
 * idxc. The routine will perform optimal contraction using symmetry and
 * sparsity information obtained from tensors' block structures. It is the
 * user's responsibility to setup all tensors so that they have correct
 * symmetries. This function does not change the original symmetry of the
 * resulting tensor c.
 *
 * Example: xm_contract(1.0, vvvv, oovv, 0.0, t2, "abcd", "ijcd", "ijab");
 */
void xm_contract(xm_scalar_t alpha, xm_tensor_t *a, xm_tensor_t *b,
    xm_scalar_t beta, xm_tensor_t *c, const char *idxa, const char *idxb,
    const char *idxc);

/* Compute y = alpha * x + y */
void xm_axpy(xm_scalar_t alpha, xm_tensor_t *x, xm_tensor_t *y,
    const char *xidx, const char *yidx);


/* Operations on block-spaces. */

/* Create a block-space with specific absolute dimensions. */
xm_block_space_t *xm_block_space_create(const xm_dim_t *dims);

/* Create a deep copy of a block-space. */
xm_block_space_t *xm_block_space_clone(const xm_block_space_t *bs);

/* Return number of dimensions a block-space has. */
size_t xm_block_space_get_ndims(const xm_block_space_t *bs);

/* Return absolute dimensions of a block-space. */
xm_dim_t xm_block_space_get_abs_dims(const xm_block_space_t *bs);

/* Return block-space dimensions in number of blocks. */
xm_dim_t xm_block_space_get_nblocks(const xm_block_space_t *bs);

/* Split block-space along a dimension at point x. */
void xm_block_space_split(xm_block_space_t *bs, size_t dim, size_t x);

/* Return dimensions of a block with specific index. */
xm_dim_t xm_block_space_get_block_dims(const xm_block_space_t *bs,
    const xm_dim_t *blkidx);

/* Return size in number of elements of the largest block in block-space. */
size_t xm_block_space_get_largest_block_size(const xm_block_space_t *bs);

/* Return non-zero if the block-spaces have same block structures. */
int xm_block_space_eq(const xm_block_space_t *bsa, const xm_block_space_t *bsb);

/* Return non-zero if specific block-space dimensions have same block
 * structures. */
int xm_block_space_eq1(const xm_block_space_t *bsa, size_t dima,
    const xm_block_space_t *bsb, size_t dimb);

/* Release resources used by a block-space. */
void xm_block_space_free(xm_block_space_t *bs);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_H_INCLUDED */
