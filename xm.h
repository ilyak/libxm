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

/* Maximum number of tensor dimensions; change if necessary. */
#define XM_MAX_DIM 6

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

/* A multi-dimensional block-space. */
typedef struct xm_block_space xm_block_space_t;

/* Multidimensional tensor index. */
typedef struct {
	size_t n, i[XM_MAX_DIM];
} xm_dim_t;

/* Print libxm banner to standard output. */
void xm_print_banner(void);

/* Initialize all indices of a dim to zero. */
xm_dim_t xm_dim_zero(size_t n);

/* Initialize all indices of a dim to the same value. */
xm_dim_t xm_dim_same(size_t n, size_t dim);

/* Initialize a 2-D dim. */
xm_dim_t xm_dim_2(size_t dim1, size_t dim2);

/* Initialize a 3-D dim. */
xm_dim_t xm_dim_3(size_t dim1, size_t dim2, size_t dim3);

/* Initialize a 4-D dim. */
xm_dim_t xm_dim_4(size_t dim1, size_t dim2, size_t dim3, size_t dim4);

/* Initialize a 5-D dim. */
xm_dim_t xm_dim_5(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5);

/* Initialize a 6-D dim. */
xm_dim_t xm_dim_6(size_t dim1, size_t dim2, size_t dim3, size_t dim4,
    size_t dim5, size_t dim6);

/* Returns an n-dimensional identity permutation. */
xm_dim_t xm_dim_identity_permutation(size_t n);

/* Multiply all dimensions of a dim by value s. */
xm_dim_t xm_dim_scale(const xm_dim_t *dim, size_t s);

/* Returns dot product of all indices of a dim. */
size_t xm_dim_dot(const xm_dim_t *dim);

/* Returns non-zero if index is within zero and dim. */
int xm_dim_less(const xm_dim_t *idx, const xm_dim_t *dim);

/* Increment an index by one wrapping on dimensions.
 * Returns nonzero if the operation has wrapped to an all-zero index. */
size_t xm_dim_inc(xm_dim_t *idx, const xm_dim_t *dim);

xm_block_space_t *xm_block_space_create(const xm_dim_t *);
size_t xm_block_space_get_ndims(const xm_block_space_t *);
xm_dim_t xm_block_space_get_abs_dims(const xm_block_space_t *);
xm_dim_t xm_block_space_get_nblocks(const xm_block_space_t *);
void xm_block_space_split(xm_block_space_t *, size_t, size_t);
xm_dim_t xm_block_space_get_block_dims(const xm_block_space_t *,
    const xm_dim_t *);
int xm_block_space_eq(const xm_block_space_t *, const xm_block_space_t *);
int xm_block_space_eq1(const xm_block_space_t *, size_t,
    const xm_block_space_t *, size_t);
void xm_block_space_free(xm_block_space_t *);

/* Create a labeled tensor specifying its dimensions in blocks. */
xm_tensor_t *xm_tensor_create(xm_block_space_t *bs, const char *label,
    xm_allocator_t *allocator);

/* Returns an allocator associated with this tensor. */
xm_allocator_t *xm_tensor_get_allocator(xm_tensor_t *tensor);

/* Returns tensor label. */
const char *xm_tensor_get_label(const xm_tensor_t *tensor);

/* Copy tensor data. Tensors must have exactly the same block structure. */
void xm_tensor_copy_data(xm_tensor_t *dst, const xm_tensor_t *src);

/* Returns tensor dimensions in number of blocks. */
xm_dim_t xm_tensor_get_nblocks(const xm_tensor_t *tensor);

/* Returns absolute tensor dimensions in total number of elements. */
xm_dim_t xm_tensor_get_abs_dims(const xm_tensor_t *tensor);

/* Returns an individual tensor element given block index and element index
 * within a block. Note: this function is slow. */
xm_scalar_t xm_tensor_get_element(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx, const xm_dim_t *el_idx);

/* Returns an individual element of a tensor given its absolute index.
 * Note: this function is slow. */
xm_scalar_t xm_tensor_get_abs_element(const xm_tensor_t *tensor,
    const xm_dim_t *idx);

/* Check if the block is non-zero. */
int xm_tensor_block_is_nonzero(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx);

/* Get block dimensions. */
xm_dim_t xm_tensor_get_block_dims(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx);

/* Get block data pointer. */
uintptr_t xm_tensor_get_block_data_ptr(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx);

/* Get permutation of a block. */
xm_dim_t xm_tensor_get_block_permutation(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx);

/* Get scalar multiplier for a block. */
xm_scalar_t xm_tensor_get_block_scalar(const xm_tensor_t *tensor,
    const xm_dim_t *blk_idx);

/* Reset block to the uninitialized state. This does not deallocate memory
 * allocated for the specified block. */
void xm_tensor_reset_block(xm_tensor_t *tensor, const xm_dim_t *blk_idx);

/* Set block as zero-block. No actual data is stored. */
void xm_tensor_set_zero_block(xm_tensor_t *tensor, const xm_dim_t *blk_idx);

/* Setup the source (canonical) block.
 * Each unique source block must be setup using this function before being used
 * in xm_tensor_set_block.
 * Note: if blocks are allocated using a disk-backed allocator they should be
 * at least several megabytes in size for best performance (e.g., 32^4 elements
 * for 4-index tensors).
 * The data_ptr argument must be allocated using the same allocator that was
 * used during tensor creation. It must be large enough to hold block data. */
void xm_tensor_set_source_block(xm_tensor_t *tensor, const xm_dim_t *blk_idx,
    uintptr_t data_ptr);

/* Allocate storage sufficient to hold data for a particular block. */
uintptr_t xm_allocate_block_data(xm_tensor_t *tensor, const xm_dim_t *blk_idx);

/* Setup a non-canonical block.
 * A non-canonical block is a copy of some source block with applied
 * permutation and multiplication by a scalar factor.
 * No actual data is stored for non-canonical blocks. */
void xm_tensor_set_block(xm_tensor_t *tensor, const xm_dim_t *blk_idx,
    const xm_dim_t *source_idx, const xm_dim_t *perm, xm_scalar_t scalar);

/* Deallocate all blocks associated with this tensor. */
void xm_tensor_free_blocks(xm_tensor_t *tensor);

/* Release a tensor. The actual block-data is not freed by this function.
 * xm_tensor_free_blocks can be used to deallocate those data. */
void xm_tensor_free(xm_tensor_t *tensor);

/* Contract two tensors (c = alpha * a * b + beta * c) over contraction indices
 * specified by strings idxa and idxb. Permutation of tensor c is specified by
 * idxc. The routine will perform optimal contraction using symmetry and
 * sparsity information obtained from tensors' block-structures. It is user's
 * responsibility to setup all tensors so that they have correct symmetries and
 * block-structures.
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
