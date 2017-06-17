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

#ifndef TENSOR_H_INCLUDED
#define TENSOR_H_INCLUDED

#include "alloc.h"
#include "blockspace.h"

#ifdef __cplusplus
extern "C" {
#endif

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

/* Create a block-tensor. */
xm_tensor_t *xm_tensor_create(const xm_block_space_t *bs,
    xm_allocator_t *allocator);

/* Clone a tensor and all its block-data using given allocator. If allocator
 * argument is NULL, use the same allocator as the tensor being cloned. */
xm_tensor_t *xm_tensor_clone(const xm_tensor_t *tensor,
    xm_allocator_t *allocator);

/* Copy tensor block data from src to dst. Tensors must have identical
 * block-structures. */
void xm_tensor_copy(xm_tensor_t *dst, const xm_tensor_t *src);

/* Scale the tensor. */
void xm_tensor_scale(xm_tensor_t *tensor, xm_scalar_t s);

/* Return block-space associated with this tensor. */
const xm_block_space_t *xm_tensor_get_block_space(const xm_tensor_t *tensor);

/* Return allocator associated with this tensor. */
xm_allocator_t *xm_tensor_get_allocator(xm_tensor_t *tensor);

/* Return absolute tensor dimensions in total number of elements. */
xm_dim_t xm_tensor_get_abs_dims(const xm_tensor_t *tensor);

/* Return tensor dimensions in number of blocks. */
xm_dim_t xm_tensor_get_nblocks(const xm_tensor_t *tensor);

/* Return an individual element of a tensor given its absolute index.
 * Note: this function is relatively slow. */
xm_scalar_t xm_tensor_get_element(const xm_tensor_t *tensor, xm_dim_t idx);

/* Return type of a block. This returns one of the XM_BLOCK_TYPE_ values. */
int xm_tensor_get_block_type(const xm_tensor_t *tensor, xm_dim_t blkidx);

/* Return dimensions of a specific block. */
xm_dim_t xm_tensor_get_block_dims(const xm_tensor_t *tensor, xm_dim_t blkidx);

/* Return size in number of elements of a specific tensor block. */
size_t xm_tensor_get_block_size(const xm_tensor_t *tensor, xm_dim_t blkidx);

/* Return block data pointer. This returns XM_NULL_PTR for zero blocks. */
uintptr_t xm_tensor_get_block_data_ptr(const xm_tensor_t *tensor,
    xm_dim_t blkidx);

/* Return permutation of a specific tensor block. */
xm_dim_t xm_tensor_get_block_permutation(const xm_tensor_t *tensor,
    xm_dim_t blkidx);

/* Return scalar factor of a specific tensor block. */
xm_scalar_t xm_tensor_get_block_scalar(const xm_tensor_t *tensor,
    xm_dim_t blkidx);

/* Set tensor block as zero-block (all elements of a block are zeros).
 * No actual data are stored for zero-blocks. */
void xm_tensor_set_zero_block(xm_tensor_t *tensor, xm_dim_t blkidx);

/* Set tensor block as canonical block. Canonical blocks are the only ones that
 * store actual data.
 * Note: if blocks are allocated using a disk-backed allocator they should be
 * at least several megabytes in size for best performance (e.g., 32^4 elements
 * for 4-index tensors).
 * The data_ptr argument must be allocated using the same allocator that was
 * used during tensor creation. Allocation must be large enough to hold block
 * data. See also: xm_tensor_allocate_block_data. */
void xm_tensor_set_canonical_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    uintptr_t data_ptr);

/* Set tensor block as derivative block. A derivative block is a copy of some
 * canonical block with applied permutation and multiplication by a scalar
 * factor. No actual data are stored for derivative blocks. */
void xm_tensor_set_derivative_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t source_blkidx, xm_dim_t permutation, xm_scalar_t scalar);

/* Unfold block into the matrix form. The sequences of unfolding indices are
 * specified using the masks. The from parameter should point to the raw block
 * data in memory. The stride must be equal to or greater than the product of
 * mask_i block dimensions. */
void xm_tensor_unfold_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t mask_i, xm_dim_t mask_j, const xm_scalar_t *from,
    xm_scalar_t *to, size_t stride);

/* Fold block back from the matrix form. This is the inverse of the
 * xm_tensor_unfold_block function. On return, "to" will contain raw block data
 * that can be directly written to the data_ptr of the block. Only canonical
 * blocks can be folded. */
void xm_tensor_fold_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t mask_i, xm_dim_t mask_j, const xm_scalar_t *from,
    xm_scalar_t *to, size_t stride);

/* Allocate storage sufficient to hold data for a particular block using
 * associated allocator. */
uintptr_t xm_tensor_allocate_block_data(xm_tensor_t *tensor, xm_dim_t blkidx);

/* Deallocate associated data for all blocks of this tensor. */
void xm_tensor_free_block_data(xm_tensor_t *tensor);

/* Release resources associated with a tensor. The actual block data are not
 * freed by this function. Use xm_tensor_free_block_data to do it. */
void xm_tensor_free(xm_tensor_t *tensor);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TENSOR_H_INCLUDED */
