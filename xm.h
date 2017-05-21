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

/* Print libxm banner to the standard output. */
void xm_print_banner(void);

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
xm_scalar_t xm_tensor_get_element(const xm_tensor_t *tensor, xm_dim_t idx);

/* Get elements of a particular block. The storage pointed to by data should
 * be large enough to store all elements of a block. */
void xm_tensor_get_block_elements(xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_scalar_t *data, size_t data_bytes);

/* Return type of a block. This returns one of the XM_BLOCK_TYPE_ values. */
int xm_tensor_get_block_type(const xm_tensor_t *tensor, xm_dim_t blkidx);

/* Return dimensions of a specific block. */
xm_dim_t xm_tensor_get_block_dims(const xm_tensor_t *tensor, xm_dim_t blkidx);

/* Setup a zero-block (all elements of a block are zeros).
 * No actual data are stored. */
void xm_tensor_set_zero_block(xm_tensor_t *tensor, xm_dim_t blkidx);

/* Setup a canonical tensor block. Canonical blocks are the only ones that
 * store actual data.
 * Note: if blocks are allocated using a disk-backed allocator they should be
 * at least several megabytes in size for best performance (e.g., 32^4 elements
 * for 4-index tensors).
 * The data_ptr argument must be allocated using the same allocator that was
 * used during tensor creation. Allocation must be large enough to hold block
 * data. */
void xm_tensor_set_canonical_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    uintptr_t data_ptr);

/* Setup a derivative block. A derivative block is a copy of some canonical
 * block with applied permutation and multiplication by a scalar factor.
 * No actual data are stored for derivative blocks. */
void xm_tensor_set_derivative_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t source_blkidx, xm_dim_t permutation, xm_scalar_t scalar);

/* Allocate storage sufficient to hold data for a particular block using
 * associated allocator. */
uintptr_t xm_tensor_allocate_block_data(xm_tensor_t *tensor, xm_dim_t blkidx);

/* Return block data pointer. This will return XM_NULL_PTR for zero and
 * derivative blocks. */
uintptr_t xm_tensor_get_block_data_ptr(xm_tensor_t *tensor, xm_dim_t blkidx);

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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_H_INCLUDED */
