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

/* Return non-zero if an index is within zero and dim. */
int xm_dim_less(const xm_dim_t *idx, const xm_dim_t *dim);

/* Increment an index by one wrapping on dimensions.
 * Returns non-zero if the operation has wrapped to an all-zero index. */
size_t xm_dim_inc(xm_dim_t *idx, const xm_dim_t *dim);


/* Operations on block-spaces. */

/* Create a block-space with specific absolute dimensions. */
xm_block_space_t *xm_block_space_create(const xm_dim_t *dims);

/* Create a deep copy of a block-space. */
xm_block_space_t *xm_block_space_clone(const xm_block_space_t *bs);

/* Return number of dimensions of a block-space. */
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


/* Operations on tensors. */

/* Create a block-tensor. */
xm_tensor_t *xm_tensor_create(const xm_block_space_t *bs,
    xm_allocator_t *allocator);

/* Return a block-space associated with this tensor. */
const xm_block_space_t *xm_tensor_get_block_space(const xm_tensor_t *tensor);

/* Return an allocator associated with this tensor. */
xm_allocator_t *xm_tensor_get_allocator(xm_tensor_t *tensor);

/* Copy tensor block data from src to dst.
 * Tensors must have exactly the same block structures. Blocks must be
 * allocated beforehand in the destination tensor. */
void xm_tensor_copy_data(xm_tensor_t *dst, const xm_tensor_t *src);

/* Return tensor dimensions in number of blocks. */
xm_dim_t xm_tensor_get_nblocks(const xm_tensor_t *tensor);

/* Return absolute tensor dimensions in total number of elements. */
xm_dim_t xm_tensor_get_abs_dims(const xm_tensor_t *tensor);

/* Return an individual tensor element given block index and element index
 * within a block. Note: this function is slow. */
xm_scalar_t xm_tensor_get_element(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx, const xm_dim_t *el_idx);

/* Return an individual element of a tensor given its absolute index.
 * Note: this function is slow. */
xm_scalar_t xm_tensor_get_abs_element(const xm_tensor_t *tensor,
    const xm_dim_t *idx);

/* Return type of a block. This returns one of the XM_BLOCK_TYPE_ values. */
int xm_tensor_get_block_type(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx);

/* Return dimensions of a specific block. */
xm_dim_t xm_tensor_get_block_dims(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx);

/* Return block data pointer. */
uintptr_t xm_tensor_get_block_data_ptr(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx);

/* Return permutation for a block. */
xm_dim_t xm_tensor_get_block_permutation(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx);

/* Return scalar multiplier for a block. */
xm_scalar_t xm_tensor_get_block_scalar(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx);

/* Allocate storage sufficient to hold data for a particular block. */
uintptr_t xm_tensor_allocate_block_data(xm_tensor_t *tensor,
    const xm_dim_t *blk_idx);

/* Setup a zero-block (all elements of a block are zeros).
 * No actual data are stored. */
void xm_tensor_set_zero_block(xm_tensor_t *tensor, const xm_dim_t *blk_idx);

/* Setup a canonical block. Each canonical block must be initialized using this
 * function before being used in xm_tensor_set_derivative_block.
 * Note: if blocks are allocated using a disk-backed allocator they should be
 * at least several megabytes in size for best performance (e.g., 32^4 elements
 * for 4-index tensors).
 * The data_ptr argument must be allocated using the same allocator that was
 * used during tensor creation. Allocation must be large enough to hold block
 * data. */
void xm_tensor_set_canonical_block(xm_tensor_t *tensor, const xm_dim_t *blk_idx,
    uintptr_t data_ptr);

/* Setup a derivative block. A derivative block is a copy of some canonical
 * block with applied permutation and multiplication by a scalar factor.
 * No actual data are stored for derivative blocks. */
void xm_tensor_set_derivative_block(xm_tensor_t *tensor, const xm_dim_t *blk_idx, const xm_dim_t *source_idx, const xm_dim_t *permutation, xm_scalar_t scalar);

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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_H_INCLUDED */
