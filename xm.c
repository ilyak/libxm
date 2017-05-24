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

#include <assert.h>
#include <ctype.h>
#include <err.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xm.h"

#define BATCH_BLOCKS_K 128

/** defines a tensor block */
struct xm_block {
	/** pointer to data allocated with an allocator */
	uintptr_t               data_ptr;
	/** block dimensions in scalar elements */
	xm_dim_t                dim;
	/** permutation that is applied to raw data */
	xm_dim_t                permutation;
	/** index of the canonical block if this block is not a source block */
	xm_dim_t                source_blkidx;
	/** scalar multiplier for the raw data */
	xm_scalar_t             scalar;
	/** type of this block */
	int                     type;
};

struct xm_tensor {
	xm_block_space_t *bs;
	xm_allocator_t *allocator;
	struct xm_block *blocks;
};

struct xm_ctx {
	xm_scalar_t alpha, beta;
	xm_tensor_t *a, *b, *c;
	xm_dim_t cidxa, aidxa;
	xm_dim_t cidxb, aidxb;
	xm_dim_t cidxc, aidxc;
	size_t nblk_m, nblk_n, nblk_k;
};

#if defined(XM_SCALAR_FLOAT)
#define xm_blas_gemm sgemm_
#elif defined(XM_SCALAR_DOUBLE_COMPLEX)
#define xm_blas_gemm zgemm_
#elif defined(XM_SCALAR_FLOAT_COMPLEX)
#define xm_blas_gemm cgemm_
#else /* assume double */
#define xm_blas_gemm dgemm_
#endif

void xm_blas_gemm(char *, char *, int *, int *, int *, xm_scalar_t *,
    xm_scalar_t *, int *, xm_scalar_t *, int *, xm_scalar_t *,
    xm_scalar_t *, int *);

static void
gemm_wrapper(char transa, char transb, int m, int n, int k, xm_scalar_t alpha,
    xm_scalar_t *a, int lda, xm_scalar_t *b, int ldb, xm_scalar_t beta,
    xm_scalar_t *c, int ldc)
{
	xm_blas_gemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda,
	    b, &ldb, &beta, c, &ldc);
}

static void *
xmalloc(size_t size)
{
	void *ptr;

	if (size == 0)
		errx(1, "xmalloc: zero size");
	ptr = malloc(size);
	if (ptr == NULL)
		errx(1, "xmalloc: allocating %zu bytes: %s",
		    size, strerror(errno));
	return ptr;
}

static struct xm_block *
xm_tensor_get_block(const xm_tensor_t *tensor, const xm_dim_t *idx)
{
	xm_dim_t nblocks;
	size_t offset;

	assert(tensor);
	assert(idx);

	nblocks = xm_tensor_get_nblocks(tensor);
	offset = xm_dim_offset(idx, &nblocks);
	return (tensor->blocks + offset);
}

xm_tensor_t *
xm_tensor_create(const xm_block_space_t *bs, xm_allocator_t *allocator)
{
	xm_tensor_t *tensor;
	xm_dim_t idx, nblocks;

	assert(bs);
	assert(allocator);

	if ((tensor = calloc(1, sizeof *tensor)) == NULL)
		return NULL;
	nblocks = xm_block_space_get_nblocks(bs);
	tensor->blocks = calloc(xm_dim_dot(&nblocks), sizeof *tensor->blocks);
	if (tensor->blocks == NULL) {
		xm_tensor_free(tensor);
		return NULL;
	}
	if ((tensor->bs = xm_block_space_clone(bs)) == NULL) {
		xm_tensor_free(tensor);
		return NULL;
	}
	tensor->allocator = allocator;
	for (idx = xm_dim_zero(nblocks.n);
	     xm_dim_ne(&idx, &nblocks);
	     xm_dim_inc(&idx, &nblocks))
		xm_tensor_set_zero_block(tensor, idx);
	return (tensor);
}

xm_tensor_t *
xm_tensor_clone(const xm_tensor_t *tensor, xm_allocator_t *allocator)
{
	xm_tensor_t *ret;
	struct xm_block *srcblk, *dstblk;
	xm_dim_t idx, blkdims, nblocks;
	xm_scalar_t *data;
	size_t blksize, maxblksize;

	if (allocator == NULL)
		allocator = tensor->allocator;
	if ((ret = xm_tensor_create(tensor->bs, allocator)) == NULL)
		return NULL;
	nblocks = xm_block_space_get_nblocks(ret->bs);
	maxblksize = xm_block_space_get_largest_block_size(ret->bs);
	data = xmalloc(maxblksize * sizeof(xm_scalar_t));
	for (idx = xm_dim_zero(nblocks.n);
	     xm_dim_ne(&idx, &nblocks);
	     xm_dim_inc(&idx, &nblocks)) {
		srcblk = xm_tensor_get_block(tensor, &idx);
		dstblk = xm_tensor_get_block(ret, &idx);
		*dstblk = *srcblk;
		dstblk->data_ptr = XM_NULL_PTR;
		if (dstblk->type != XM_BLOCK_TYPE_CANONICAL)
			continue;
		blkdims = xm_block_space_get_block_dims(ret->bs, idx);
		blksize = xm_dim_dot(&blkdims);
		dstblk->data_ptr = xm_allocator_allocate(ret->allocator,
		    blksize * sizeof(xm_scalar_t));
		if (dstblk->data_ptr == XM_NULL_PTR) {
			xm_tensor_free_block_data(ret);
			xm_tensor_free(ret);
			free(data);
			return NULL;
		}
		xm_allocator_read(tensor->allocator, srcblk->data_ptr,
		    data, blksize * sizeof(xm_scalar_t));
		xm_allocator_write(ret->allocator, dstblk->data_ptr,
		    data, blksize * sizeof(xm_scalar_t));
	}
	free(data);
	return ret;
}

const xm_block_space_t *
xm_tensor_get_block_space(const xm_tensor_t *tensor)
{
	return (tensor->bs);
}

xm_allocator_t *
xm_tensor_get_allocator(xm_tensor_t *tensor)
{
	return (tensor->allocator);
}

static uintptr_t
get_source_block_data_ptr(xm_tensor_t *tensor, struct xm_block *block)
{
	if (block->type == XM_BLOCK_TYPE_CANONICAL)
		return block->data_ptr;
	if (block->type == XM_BLOCK_TYPE_DERIVATIVE) {
		struct xm_block *source;
		source = xm_tensor_get_block(tensor, &block->source_blkidx);
		return source->data_ptr;
	}
	return XM_NULL_PTR;
}

//void
//xm_tensor_copy_data(xm_tensor_t *dst, const xm_tensor_t *src)
//{
//	struct xm_block *dstblk, *srcblk;
//	xm_dim_t nblocks;
//	xm_scalar_t *buf;
//	size_t i, nblk, size, maxblocksize;
//
//	assert(xm_block_space_eq(dst->bs, src->bs));
//
//	nblocks = xm_tensor_get_nblocks(dst);
//	nblk = xm_dim_dot(&nblocks);
//	maxblocksize = xm_block_space_get_largest_block_size(src->bs);
//	buf = xmalloc(maxblocksize * sizeof *buf);
//
//	for (i = 0; i < nblk; i++) {
//		dstblk = &dst->blocks[i];
//		srcblk = &src->blocks[i];
//		assert(srcblk->type == dstblk->type);
//		assert(xm_dim_eq(&srcblk->dim, &dstblk->dim));
//		if (dstblk->type == XM_BLOCK_TYPE_CANONICAL) {
//			size = xm_dim_dot(&dstblk->dim) * sizeof(xm_scalar_t);
//			assert(srcblk->data_ptr != XM_NULL_PTR);
//			xm_allocator_read(src->allocator, srcblk->data_ptr,
//			    buf, size);
//			assert(dstblk->data_ptr != XM_NULL_PTR);
//			xm_allocator_write(dst->allocator, dstblk->data_ptr,
//			    buf, size);
//		}
//	}
//	free(buf);
//}

static void
xm_tensor_get_idx(const xm_tensor_t *tensor, const xm_dim_t *abs_idx,
    xm_dim_t *blk_idx, xm_dim_t *el_idx)
{
	struct xm_block *block;
	xm_dim_t abs_dim, idx, nblocks;
	size_t dim_i, blk_i, next;

	nblocks = xm_tensor_get_nblocks(tensor);
	*blk_idx = xm_dim_zero(nblocks.n);
	*el_idx = xm_dim_zero(nblocks.n);

	abs_dim = xm_dim_zero(nblocks.n);
	for (dim_i = 0; dim_i < nblocks.n; dim_i++) {
		idx = xm_dim_zero(nblocks.n);
		for (blk_i = 0; blk_i < nblocks.i[dim_i]; blk_i++) {
			idx.i[dim_i] = blk_i;
			block = xm_tensor_get_block(tensor, &idx);
			next = abs_dim.i[dim_i] + block->dim.i[dim_i];
			if (next > abs_idx->i[dim_i]) {
				el_idx->i[dim_i] = abs_idx->i[dim_i] -
				    abs_dim.i[dim_i];
				break;
			}
			abs_dim.i[dim_i] = next;
			blk_idx->i[dim_i]++;
		}
		if (blk_i == nblocks.i[dim_i])
			assert(0);
	}
}

xm_scalar_t
xm_tensor_get_element(const xm_tensor_t *tensor, xm_dim_t idx)
{
	struct xm_block *block;
	xm_scalar_t *buf, ret;
	xm_dim_t eidx, dim_p, blk_i, el_i;
	size_t el_off, size_bytes;
	uintptr_t data_ptr = XM_NULL_PTR;

	xm_tensor_get_idx(tensor, &idx, &blk_i, &el_i);
	block = xm_tensor_get_block(tensor, &blk_i);
	if (block->type == XM_BLOCK_TYPE_ZERO)
		return (0.0);
	if (block->type == XM_BLOCK_TYPE_CANONICAL)
		data_ptr = block->data_ptr;
	if (block->type == XM_BLOCK_TYPE_DERIVATIVE) {
		struct xm_block *canon_block = xm_tensor_get_block(tensor,
		    &block->source_blkidx);
		data_ptr = canon_block->data_ptr;
	}
	assert(data_ptr != XM_NULL_PTR);
	size_bytes = xm_dim_dot(&block->dim) * sizeof(xm_scalar_t);
	buf = xmalloc(size_bytes);
	xm_allocator_read(tensor->allocator, data_ptr, buf, size_bytes);
	eidx = xm_dim_permute(&el_i, &block->permutation);
	dim_p = xm_dim_permute(&block->dim, &block->permutation);
	el_off = xm_dim_offset(&eidx, &dim_p);
	ret = block->scalar * buf[el_off];
	free(buf);
	return (ret);
}

int
xm_tensor_get_block_type(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, &blkidx);
	return (block->type);
}

xm_dim_t
xm_tensor_get_block_dims(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	assert(tensor);
	return xm_block_space_get_block_dims(tensor->bs, blkidx);
}

uintptr_t
xm_tensor_allocate_block_data(xm_tensor_t *tensor, xm_dim_t blkidx)
{
	xm_dim_t blkdims;
	size_t size;

	assert(tensor);

	blkdims = xm_tensor_get_block_dims(tensor, blkidx);
	size = xm_dim_dot(&blkdims) * sizeof(xm_scalar_t);

	return xm_allocator_allocate(tensor->allocator, size);
}

uintptr_t
xm_tensor_get_block_data_ptr(xm_tensor_t *tensor, xm_dim_t blkidx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, &blkidx);
	return (block->data_ptr);
}

//xm_dim_t
//xm_tensor_get_block_permutation(const xm_tensor_t *tensor, xm_dim_t blkidx)
//{
//	struct xm_block *block;
//
//	block = xm_tensor_get_block(tensor, &blkidx);
//	return (block->permutation);
//}
//
//xm_scalar_t
//xm_tensor_get_block_scalar(const xm_tensor_t *tensor, const xm_dim_t *idx)
//{
//	struct xm_block *block;
//
//	block = xm_tensor_get_block(tensor, idx);
//	return (block->scalar);
//}

void
xm_tensor_set_zero_block(xm_tensor_t *tensor, xm_dim_t blkidx)
{
	struct xm_block *block;
	xm_dim_t blkdims;

	assert(tensor != NULL);

	blkdims = xm_block_space_get_block_dims(tensor->bs, blkidx);
	block = xm_tensor_get_block(tensor, &blkidx);
	block->source_blkidx = blkidx;
	block->dim = blkdims;
	block->data_ptr = XM_NULL_PTR;
	block->permutation = xm_dim_identity_permutation(blkdims.n);
	block->scalar = 1.0;
	block->type = XM_BLOCK_TYPE_ZERO;
}

void
xm_tensor_set_canonical_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    uintptr_t data_ptr)
{
	struct xm_block *block;
	xm_dim_t blkdims;

	assert(tensor != NULL);
	assert(data_ptr != XM_NULL_PTR);

	blkdims = xm_block_space_get_block_dims(tensor->bs, blkidx);
	block = xm_tensor_get_block(tensor, &blkidx);
	block->source_blkidx = blkidx;
	block->dim = blkdims;
	block->data_ptr = data_ptr;
	block->permutation = xm_dim_identity_permutation(blkdims.n);
	block->scalar = 1.0;
	block->type = XM_BLOCK_TYPE_CANONICAL;
}

void
xm_tensor_set_derivative_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t source_blkidx, xm_dim_t permutation, xm_scalar_t scalar)
{
	struct xm_block *block, *source_block;
	xm_dim_t blkdim, bsdim;

	assert(tensor != NULL);

	source_block = xm_tensor_get_block(tensor, &source_blkidx);
	assert(source_block->type == XM_BLOCK_TYPE_CANONICAL);

	block = xm_tensor_get_block(tensor, &blkidx);
	bsdim = xm_block_space_get_block_dims(tensor->bs, blkidx);
	blkdim = xm_dim_permute(&bsdim, &permutation);
	assert(xm_dim_eq(&blkdim, &source_block->dim));
	block->source_blkidx = source_blkidx;
	block->dim = bsdim;
	block->data_ptr = XM_NULL_PTR;
	block->permutation = permutation;
	block->scalar = scalar;
	block->type = XM_BLOCK_TYPE_DERIVATIVE;
}

xm_dim_t
xm_tensor_get_nblocks(const xm_tensor_t *tensor)
{
	assert(tensor);
	return xm_block_space_get_nblocks(tensor->bs);
}

xm_dim_t
xm_tensor_get_abs_dims(const xm_tensor_t *tensor)
{
	assert(tensor);
	return xm_block_space_get_abs_dims(tensor->bs);
}

void
xm_tensor_free_block_data(xm_tensor_t *tensor)
{
	xm_dim_t nblocks;
	size_t i, nblk;

	assert(tensor);
	nblocks = xm_tensor_get_nblocks(tensor);
	nblk = xm_dim_dot(&nblocks);
	for (i = 0; i < nblk; i++)
		if (tensor->blocks[i].type == XM_BLOCK_TYPE_CANONICAL) {
			xm_allocator_deallocate(tensor->allocator,
			    tensor->blocks[i].data_ptr);
		}
}

void
xm_tensor_free(xm_tensor_t *tensor)
{
	if (tensor) {
		xm_block_space_free(tensor->bs);
		free(tensor->blocks);
		free(tensor);
	}
}

static void
block_get_matrix(struct xm_block *block, xm_dim_t mask_i, xm_dim_t mask_j,
    size_t block_size_i, size_t block_size_j, xm_scalar_t *from,
    xm_scalar_t *to, size_t stride)
{
	xm_dim_t el_dim, el_dim_p, el_i, idx, permutation;
	size_t ii, jj, offset, inc, lead_ii, kk, el_dim_lead_ii;
	xm_scalar_t scalar = block->scalar;

	assert(from);
	assert(to);

	el_dim = block->dim;
	permutation = block->permutation;
	el_dim_p = xm_dim_permute(&block->dim, &permutation);
	el_i = xm_dim_zero(el_dim.n);

	inc = 1;
	el_dim_lead_ii = 1;

	if (mask_i.n > 0) {
		lead_ii = mask_i.i[0];
		for (kk = 0; kk < permutation.i[lead_ii]; kk++)
			inc *= el_dim_p.i[kk];
		for (ii = 0; ii < mask_i.n-1; ii++)
			mask_i.i[ii] = mask_i.i[ii+1];
		mask_i.n--;
		el_dim_lead_ii = el_dim.i[lead_ii];
	}
	if (inc == 1) {
		for (jj = 0; jj < block_size_j; jj++) {
			xm_dim_zero_mask(&el_i, &mask_i);
			for (ii = 0; ii < block_size_i;
			    ii += el_dim_lead_ii) {
				idx = xm_dim_permute(&el_i, &permutation);
				offset = xm_dim_offset(&idx, &el_dim_p);
				memcpy(&to[jj * stride + ii],
				    from + offset,
				    sizeof(xm_scalar_t) * el_dim_lead_ii);
				xm_dim_inc_mask(&el_i, &el_dim, &mask_i);
			}
			xm_dim_inc_mask(&el_i, &el_dim, &mask_j);
		}
	} else {
		for (jj = 0; jj < block_size_j; jj++) {
			xm_dim_zero_mask(&el_i, &mask_i);
			for (ii = 0; ii < block_size_i;
			    ii += el_dim_lead_ii) {
				idx = xm_dim_permute(&el_i, &permutation);
				offset = xm_dim_offset(&idx, &el_dim_p);
				for (kk = 0; kk < el_dim_lead_ii; kk++) {
					to[jj * stride + ii + kk] =
					    from[offset];
					offset += inc;
				}
				xm_dim_inc_mask(&el_i, &el_dim, &mask_i);
			}
			xm_dim_inc_mask(&el_i, &el_dim, &mask_j);
		}
	}
	for (jj = 0; jj < block_size_j; jj++)
		for (ii = 0; ii < block_size_i; ii++)
			to[jj * stride + ii] *= scalar;
}

static void
block_set_matrix(struct xm_block *block, xm_dim_t mask_i, xm_dim_t mask_j,
    size_t block_size_i, size_t block_size_j, xm_scalar_t *data,
    size_t stride, xm_scalar_t *buf, xm_allocator_t *allocator)
{
	xm_dim_t el_dim, el_i;
	size_t ii, jj, offset;

	assert(block->type == XM_BLOCK_TYPE_CANONICAL);
	assert(block->data_ptr != XM_NULL_PTR);

	el_dim = block->dim;
	el_i = xm_dim_zero(el_dim.n);

	assert(mask_i.n > 0);
	assert(mask_i.i[0] == 0);
	for (ii = 0; ii < mask_i.n-1; ii++)
		mask_i.i[ii] = mask_i.i[ii+1];
	mask_i.n--;

	for (jj = 0; jj < block_size_j; jj++) {
		xm_dim_zero_mask(&el_i, &mask_i);
		for (ii = 0; ii < block_size_i; ii += el_dim.i[0]) {
			offset = xm_dim_offset(&el_i, &el_dim);
			memcpy(buf + offset,
			    &data[jj * stride + ii],
			    sizeof(xm_scalar_t) * el_dim.i[0]);
			xm_dim_inc_mask(&el_i, &el_dim, &mask_i);
		}
		xm_dim_inc_mask(&el_i, &el_dim, &mask_j);
	}
	xm_allocator_write(allocator, block->data_ptr, buf,
	    block_size_i * block_size_j * sizeof(xm_scalar_t));
}

static void
parse_idx(const char *str1, const char *str2, xm_dim_t *mask1, xm_dim_t *mask2)
{
	size_t i, j, len1, len2;

	mask1->n = 0;
	mask2->n = 0;
	len1 = strlen(str1);
	len2 = strlen(str2);
	for (i = 0; i < len1; i++)
		for (j = 0; j < len2; j++)
			if (str1[i] == str2[j]) {
				mask1->i[mask1->n++] = i;
				mask2->i[mask2->n++] = j;
			}
}

static int
has_k_symmetry(xm_tensor_t *a, xm_dim_t cidxa, xm_dim_t aidxa,
    size_t si1, size_t si2)
{
	struct xm_block *blk, *blk2;
	xm_dim_t idx, idx2, nblocksa;
	size_t i, j, nblk_i, nblk_j;

	nblocksa = xm_tensor_get_nblocks(a);
	if (nblocksa.i[cidxa.i[si1]] != nblocksa.i[cidxa.i[si2]])
		return (0);

	nblk_i = xm_dim_dot_mask(&nblocksa, &cidxa);
	nblk_j = xm_dim_dot_mask(&nblocksa, &aidxa);

	idx = xm_dim_zero(nblocksa.n);
	for (i = 0; i < nblk_i; i++) {
		xm_dim_zero_mask(&idx, &aidxa);
		for (j = 0; j < nblk_j; j++) {
			blk = xm_tensor_get_block(a, &idx);
			if (idx.i[cidxa.i[si1]] != idx.i[cidxa.i[si2]]) {
				idx2 = idx;
				idx2.i[cidxa.i[si1]] = idx.i[cidxa.i[si2]];
				idx2.i[cidxa.i[si2]] = idx.i[cidxa.i[si1]];
				blk2 = xm_tensor_get_block(a, &idx2);
				if (blk->type != XM_BLOCK_TYPE_ZERO ||
				    blk2->type != XM_BLOCK_TYPE_ZERO) {
					if (!xm_dim_eq(&blk->source_blkidx,
						       &blk2->source_blkidx))
						return (0);
				}
			}
			xm_dim_inc_mask(&idx, &nblocksa, &aidxa);
		}
		xm_dim_inc_mask(&idx, &nblocksa, &cidxa);
	}
	return (1);
}

static void
set_k_symmetry(xm_tensor_t *a, xm_dim_t cidxa, xm_dim_t aidxa,
    size_t si1, size_t si2, int enable)
{
	struct xm_block *blk;
	xm_dim_t idx, nblocksa;
	size_t nblk_i, nblk_j, i, j;

	nblocksa = xm_tensor_get_nblocks(a);
	nblk_i = xm_dim_dot_mask(&nblocksa, &aidxa);
	nblk_j = xm_dim_dot_mask(&nblocksa, &cidxa);

	idx = xm_dim_zero(nblocksa.n);
	for (i = 0; i < nblk_i; i++) {
		xm_dim_zero_mask(&idx, &cidxa);
		for (j = 0; j < nblk_j; j++) {
			blk = xm_tensor_get_block(a, &idx);
//			if (idx.i[cidxa.i[si1]] < idx.i[cidxa.i[si2]])
//				blk->scalar *= enable ? 2.0 : 0.5;
//			else if (idx.i[cidxa.i[si1]] > idx.i[cidxa.i[si2]]) {
//				blk->is_nonzero = !enable &&
//				    blk->data_ptr != XM_NULL_PTR;
//			}
			xm_dim_inc_mask(&idx, &nblocksa, &cidxa);
		}
		xm_dim_inc_mask(&idx, &nblocksa, &aidxa);
	}
}

static void
compute_block(struct xm_ctx *ctx, xm_dim_t blkidxc, xm_scalar_t *buf)
{
	size_t i, m, n, k = 0, nbatched = 0, stride_a, stride_b;
	xm_scalar_t *buf_a, *buf_b, *blkbuf_a, *blkbuf_b;
	xm_scalar_t *blkbuf_c1, *blkbuf_c2, *buf_a_ptr, *buf_b_ptr;
	struct xm_block *blk_c = xm_tensor_get_block(ctx->c, &blkidxc);
	xm_dim_t blkidxa, blkidxb, nblocksa, nblocksb;
	size_t maxblocksizea, maxblocksizeb, maxblocksizec;

	maxblocksizea = xm_block_space_get_largest_block_size(ctx->a->bs);
	maxblocksizeb = xm_block_space_get_largest_block_size(ctx->b->bs);
	maxblocksizec = xm_block_space_get_largest_block_size(ctx->c->bs);
	nblocksa = xm_tensor_get_nblocks(ctx->a);
	nblocksb = xm_tensor_get_nblocks(ctx->b);
	m = xm_dim_dot_mask(&blk_c->dim, &ctx->cidxc);
	n = xm_dim_dot_mask(&blk_c->dim, &ctx->aidxc);
	stride_a = BATCH_BLOCKS_K * maxblocksizea / m;
	stride_b = BATCH_BLOCKS_K * maxblocksizeb / n;
	buf_a = buf;
	buf_b = buf_a + BATCH_BLOCKS_K * maxblocksizea;
	blkbuf_a = buf_b + BATCH_BLOCKS_K * maxblocksizeb;
	blkbuf_b = blkbuf_a + maxblocksizea;
	blkbuf_c1 = blkbuf_b + maxblocksizeb;
	blkbuf_c2 = blkbuf_c1 + maxblocksizec;

	buf_a_ptr = buf_a;
	buf_b_ptr = buf_b;
	blkidxa = xm_dim_zero(xm_block_space_get_ndims(ctx->a->bs));
	blkidxb = xm_dim_zero(xm_block_space_get_ndims(ctx->b->bs));
	xm_dim_set_mask(&blkidxa, &ctx->aidxa, &blkidxc, &ctx->cidxc);
	xm_dim_set_mask(&blkidxb, &ctx->aidxb, &blkidxc, &ctx->aidxc);

	size_t blk_c_size = xm_dim_dot(&blk_c->dim);
	xm_allocator_read(ctx->c->allocator, blk_c->data_ptr,
	    blkbuf_c2, blk_c_size * sizeof(xm_scalar_t));
	xm_scalar_t scalar_save = blk_c->scalar;
	blk_c->scalar *= ctx->beta;
	if (ctx->aidxc.n > 0 && ctx->aidxc.i[0] == 0) {
		size_t mblk = xm_dim_dot_mask(&blk_c->dim, &ctx->cidxc);
		size_t nblk = xm_dim_dot_mask(&blk_c->dim, &ctx->aidxc);
		block_get_matrix(blk_c, ctx->aidxc, ctx->cidxc,
		    nblk, mblk, blkbuf_c2, blkbuf_c1, nblk);
	} else {
		size_t mblk = xm_dim_dot_mask(&blk_c->dim, &ctx->cidxc);
		size_t nblk = xm_dim_dot_mask(&blk_c->dim, &ctx->aidxc);
		block_get_matrix(blk_c, ctx->cidxc, ctx->aidxc,
		    mblk, nblk, blkbuf_c2, blkbuf_c1, mblk);
	}
	blk_c->scalar = scalar_save;

	if (ctx->alpha == 0.0)
		goto done;
	for (i = 0; i < ctx->nblk_k; i++) {
		struct xm_block *blk_a = xm_tensor_get_block(ctx->a, &blkidxa);
		struct xm_block *blk_b = xm_tensor_get_block(ctx->b, &blkidxb);

		if (blk_a->type != XM_BLOCK_TYPE_ZERO &&
		    blk_b->type != XM_BLOCK_TYPE_ZERO) {
			uintptr_t data_ptr;
			size_t mblk, nblk, kblk;
			mblk = xm_dim_dot_mask(&blk_a->dim, &ctx->aidxa);
			nblk = xm_dim_dot_mask(&blk_b->dim, &ctx->aidxb);
			kblk = xm_dim_dot_mask(&blk_a->dim, &ctx->cidxa);

			data_ptr = get_source_block_data_ptr(ctx->a, blk_a);
			size_t blk_a_size = xm_dim_dot(&blk_a->dim);
			xm_allocator_read(ctx->a->allocator, data_ptr,
			    blkbuf_a, blk_a_size * sizeof(xm_scalar_t));
			block_get_matrix(blk_a, ctx->cidxa, ctx->aidxa,
			    kblk, mblk, blkbuf_a, buf_a_ptr, stride_a);
			buf_a_ptr += kblk;

			data_ptr = get_source_block_data_ptr(ctx->b, blk_b);
			size_t blk_b_size = xm_dim_dot(&blk_b->dim);
			xm_allocator_read(ctx->b->allocator, data_ptr,
			    blkbuf_b, blk_b_size * sizeof(xm_scalar_t));
			block_get_matrix(blk_b, ctx->cidxb, ctx->aidxb,
			    kblk, nblk, blkbuf_b, buf_b_ptr, stride_b);
			buf_b_ptr += kblk;

			k += kblk;
			nbatched++;
		}
		if (nbatched >= BATCH_BLOCKS_K ||
		   (i == ctx->nblk_k-1 && nbatched > 0)) {
			if (ctx->aidxc.n > 0 && ctx->aidxc.i[0] == 0) {
				gemm_wrapper('T', 'N', (int)n, (int)m, (int)k,
				    ctx->alpha, buf_b, (int)stride_b, buf_a,
				    (int)stride_a, 1.0, blkbuf_c1, (int)n);
			} else {
				gemm_wrapper('T', 'N', (int)m, (int)n, (int)k,
				    ctx->alpha, buf_a, (int)stride_a, buf_b,
				    (int)stride_b, 1.0, blkbuf_c1, (int)m);
			}
			k = 0;
			buf_a_ptr = buf_a;
			buf_b_ptr = buf_b;
			nbatched = 0;
		}
		xm_dim_inc_mask(&blkidxa, &nblocksa, &ctx->cidxa);
		xm_dim_inc_mask(&blkidxb, &nblocksb, &ctx->cidxb);
	}
done:
	if (ctx->aidxc.n > 0 && ctx->aidxc.i[0] == 0) {
		block_set_matrix(blk_c, ctx->aidxc, ctx->cidxc, n, m,
		    blkbuf_c1, n, blkbuf_c2, ctx->c->allocator);
	} else {
		block_set_matrix(blk_c, ctx->cidxc, ctx->aidxc, m, n,
		    blkbuf_c1, m, blkbuf_c2, ctx->c->allocator);
	}
}

static xm_dim_t *
get_nonzero_blocks(struct xm_ctx *ctx, size_t *nnzblkout)
{
	struct xm_block *blk;
	xm_dim_t *nzblk, *nzblkptr, idx, nblocksc;
	size_t i, j, nnzblk = 0;

	nblocksc = xm_tensor_get_nblocks(ctx->c);
	idx = xm_dim_zero(nblocksc.n);
	for (i = 0; i < ctx->nblk_m; i++) {
		for (j = 0; j < ctx->nblk_n; j++) {
			blk = xm_tensor_get_block(ctx->c, &idx);
			if (blk->type == XM_BLOCK_TYPE_CANONICAL)
				nnzblk++;
			xm_dim_inc_mask(&idx, &nblocksc, &ctx->aidxc);
		}
		xm_dim_inc_mask(&idx, &nblocksc, &ctx->cidxc);
	}
	if ((*nnzblkout = nnzblk) == 0)
		return (NULL);
	nzblkptr = nzblk = xmalloc(nnzblk * sizeof(xm_dim_t));

	idx = xm_dim_zero(nblocksc.n);
	for (i = 0; i < ctx->nblk_m; i++) {
		for (j = 0; j < ctx->nblk_n; j++) {
			blk = xm_tensor_get_block(ctx->c, &idx);
			if (blk->type == XM_BLOCK_TYPE_CANONICAL)
				*nzblkptr++ = idx;
			xm_dim_inc_mask(&idx, &nblocksc, &ctx->aidxc);
		}
		xm_dim_inc_mask(&idx, &nblocksc, &ctx->cidxc);
	}
	return (nzblk);
}

void
xm_contract(xm_scalar_t alpha, xm_tensor_t *a, xm_tensor_t *b,
    xm_scalar_t beta, xm_tensor_t *c, const char *idxa, const char *idxb,
    const char *idxc)
{
	struct xm_ctx ctx;
	xm_dim_t cidxa, aidxa, cidxb, aidxb, cidxc, aidxc, *nzblk;
	xm_dim_t nblocksa, nblocksb;
	size_t i, si1, si2, size, nnzblk;
	size_t maxblocksizea, maxblocksizeb, maxblocksizec;
	int sym_k;

	assert(strlen(idxa) == xm_block_space_get_ndims(a->bs));
	assert(strlen(idxb) == xm_block_space_get_ndims(b->bs));
	assert(strlen(idxc) == xm_block_space_get_ndims(c->bs));

	parse_idx(idxa, idxb, &cidxa, &cidxb);
	parse_idx(idxc, idxa, &cidxc, &aidxa);
	parse_idx(idxc, idxb, &aidxc, &aidxb);

	nblocksa = xm_tensor_get_nblocks(a);
	nblocksb = xm_tensor_get_nblocks(b);

	ctx.alpha = alpha;
	ctx.beta = beta;
	ctx.a = a;
	ctx.b = b;
	ctx.c = c;
	ctx.cidxa = cidxa;
	ctx.aidxa = aidxa;
	ctx.cidxb = cidxb;
	ctx.aidxb = aidxb;
	ctx.cidxc = cidxc;
	ctx.aidxc = aidxc;
	ctx.nblk_m = xm_dim_dot_mask(&nblocksa, &aidxa);
	ctx.nblk_n = xm_dim_dot_mask(&nblocksb, &aidxb);
	ctx.nblk_k = xm_dim_dot_mask(&nblocksa, &cidxa);

	assert(aidxa.n + cidxa.n == xm_block_space_get_ndims(a->bs));
	assert(aidxb.n + cidxb.n == xm_block_space_get_ndims(b->bs));
	assert(aidxc.n + cidxc.n == xm_block_space_get_ndims(c->bs));
	assert((aidxc.n > 0 && aidxc.i[0] == 0) ||
	       (cidxc.n > 0 && cidxc.i[0] == 0));

	assert(cidxa.n == cidxb.n);
	for (i = 0; i < cidxa.n; i++) {
		assert(xm_block_space_eq1(a->bs, cidxa.i[i],
					  b->bs, cidxb.i[i]));
	}
	assert(cidxc.n == aidxa.n);
	for (i = 0; i < cidxc.n; i++) {
		assert(xm_block_space_eq1(c->bs, cidxc.i[i],
					  a->bs, aidxa.i[i]));
	}
	assert(aidxc.n == aidxb.n);
	for (i = 0; i < aidxc.n; i++) {
		assert(xm_block_space_eq1(c->bs, aidxc.i[i],
					  b->bs, aidxb.i[i]));
	}

	sym_k = 0;
	si2 = 0;
	for (si1 = 0; si1 < cidxa.n; si1++) {
		for (si2 = si1+1; si2 < cidxa.n; si2++) {
			sym_k = has_k_symmetry(a, cidxa, aidxa, si1, si2) &&
				has_k_symmetry(b, cidxb, aidxb, si1, si2);
			if (sym_k)
				break;
		}
		if (sym_k)
			break;
	}
	if (sym_k)
		set_k_symmetry(a, cidxa, aidxa, si1, si2, 1);

	maxblocksizea = xm_block_space_get_largest_block_size(a->bs);
	maxblocksizeb = xm_block_space_get_largest_block_size(b->bs);
	maxblocksizec = xm_block_space_get_largest_block_size(c->bs);
	nzblk = get_nonzero_blocks(&ctx, &nnzblk);
	size = maxblocksizea * (BATCH_BLOCKS_K + 1) +
	       maxblocksizeb * (BATCH_BLOCKS_K + 1) +
	       maxblocksizec * 2;
#ifdef _OPENMP
#pragma omp parallel private(i)
#endif
{
	xm_scalar_t *buf = xmalloc(size * sizeof(xm_scalar_t));
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < nnzblk; i++)
		compute_block(&ctx, nzblk[i], buf);
	free(buf);
}
	if (sym_k)
		set_k_symmetry(a, cidxa, aidxa, si1, si2, 0);
	free(nzblk);
}

void
xm_print_banner(void)
{
	printf("libxm (c) 2014-2017 Ilya Kaliman\n");
	printf("Code for efficient block-tensor contractions\n");
	printf("https://github.com/ilyak/libxm\n");
}
