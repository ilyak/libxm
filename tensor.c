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

#include "xm.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct xm_block {
	int type;
	xm_dim_t permutation;
	xm_scalar_t scalar;
	uintptr_t data_ptr; /* for derivative blocks stores offset of the
			       corresponding canonical block */
};

struct xm_tensor {
	xm_block_space_t *bs;
	xm_allocator_t *allocator;
	struct xm_block *blocks;
};

static void
fatal(const char *msg)
{
	fprintf(stderr, "libxm: %s\n", msg);
	abort();
}

static struct xm_block *
tensor_get_block(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	xm_dim_t nblocks;
	size_t offset;

	nblocks = xm_tensor_get_nblocks(tensor);
	offset = xm_dim_offset(&blkidx, &nblocks);

	return (&tensor->blocks[offset]);
}

void
xm_print_banner(void)
{
	printf("libxm (c) 2014-2017 Ilya Kaliman\n");
	printf("Efficient operations on block tensors\n");
	printf("https://github.com/ilyak/libxm\n");
}

xm_tensor_t *
xm_tensor_create(const xm_block_space_t *bs, xm_allocator_t *allocator)
{
	xm_dim_t idx, nblocks;
	xm_tensor_t *ret;

	assert(bs);
	assert(allocator);

	if ((ret = calloc(1, sizeof *ret)) == NULL)
		return NULL;
	if ((ret->bs = xm_block_space_clone(bs)) == NULL) {
		free(ret);
		return NULL;
	}
	ret->allocator = allocator;
	nblocks = xm_block_space_get_nblocks(bs);
	if ((ret->blocks = calloc(xm_dim_dot(&nblocks),
	    sizeof *ret->blocks)) == NULL) {
		xm_block_space_free(ret->bs);
		free(ret);
		return NULL;
	}
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		xm_tensor_set_zero_block(ret, idx);
		xm_dim_inc(&idx, &nblocks);
	}
	return ret;
}

xm_tensor_t *
xm_tensor_clone(const xm_tensor_t *tensor, xm_allocator_t *allocator)
{
	xm_tensor_t *ret;
	xm_dim_t idx, nblocks;
	xm_scalar_t *buf;
	size_t blkbytes, maxblksize;

	if (allocator == NULL)
		allocator = tensor->allocator;
	if ((ret = xm_tensor_create(tensor->bs, allocator)) == NULL)
		return NULL;
	maxblksize = xm_block_space_get_largest_block_size(ret->bs);
	if ((buf = malloc(maxblksize * sizeof(xm_scalar_t))) == NULL) {
		xm_tensor_free(ret);
		return NULL;
	}
	nblocks = xm_tensor_get_nblocks(ret);
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		size_t i = xm_dim_offset(&idx, &nblocks);
		ret->blocks[i] = tensor->blocks[i];
		if (ret->blocks[i].type == XM_BLOCK_TYPE_CANONICAL) {
			blkbytes = xm_tensor_get_block_size(ret, idx) *
			    sizeof(xm_scalar_t);
			ret->blocks[i].data_ptr = xm_allocator_allocate(
			    ret->allocator, blkbytes);
			if (ret->blocks[i].data_ptr == XM_NULL_PTR) {
				xm_tensor_free_block_data(ret);
				xm_tensor_free(ret);
				free(buf);
				return NULL;
			}
			xm_allocator_read(tensor->allocator,
			    tensor->blocks[i].data_ptr, buf, blkbytes);
			xm_allocator_write(ret->allocator,
			    ret->blocks[i].data_ptr, buf, blkbytes);
		}
		xm_dim_inc(&idx, &nblocks);
	}
	free(buf);
	return ret;
}

const xm_block_space_t *
xm_tensor_get_block_space(const xm_tensor_t *tensor)
{
	return tensor->bs;
}

xm_allocator_t *
xm_tensor_get_allocator(xm_tensor_t *tensor)
{
	return tensor->allocator;
}

xm_dim_t
xm_tensor_get_abs_dims(const xm_tensor_t *tensor)
{
	return xm_block_space_get_abs_dims(tensor->bs);
}

xm_dim_t
xm_tensor_get_nblocks(const xm_tensor_t *tensor)
{
	return xm_block_space_get_nblocks(tensor->bs);
}

xm_scalar_t
xm_tensor_get_element(const xm_tensor_t *tensor, xm_dim_t idx)
{
	struct xm_block *block;
	xm_dim_t blkidx, blkdims, elidx;
	xm_scalar_t *buf, ret;
	size_t blkbytes, eloff;
	uintptr_t data_ptr;

	xm_block_space_decompose_index(tensor->bs, idx, &blkidx, &elidx);
	block = tensor_get_block(tensor, blkidx);
	if (block->type == XM_BLOCK_TYPE_ZERO)
		return 0.0;
	data_ptr = xm_tensor_get_block_data_ptr(tensor, blkidx);
	assert(data_ptr != XM_NULL_PTR);
	blkbytes = xm_tensor_get_block_size(tensor, blkidx) *
	    sizeof(xm_scalar_t);
	if ((buf = malloc(blkbytes)) == NULL)
		fatal("out of memory");
	xm_allocator_read(tensor->allocator, data_ptr, buf, blkbytes);
	elidx = xm_dim_permute(&elidx, &block->permutation);
	blkdims = xm_tensor_get_block_dims(tensor, blkidx);
	blkdims = xm_dim_permute(&blkdims, &block->permutation);
	eloff = xm_dim_offset(&elidx, &blkdims);
	ret = block->scalar * buf[eloff];
	free(buf);
	return ret;
}

int
xm_tensor_get_block_type(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	return tensor_get_block(tensor, blkidx)->type;
}

xm_dim_t
xm_tensor_get_block_dims(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	return xm_block_space_get_block_dims(tensor->bs, blkidx);
}

size_t
xm_tensor_get_block_size(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	return xm_block_space_get_block_size(tensor->bs, blkidx);
}

void
xm_tensor_set_zero_block(xm_tensor_t *tensor, xm_dim_t blkidx)
{
	struct xm_block *block;

	block = tensor_get_block(tensor, blkidx);
	block->type = XM_BLOCK_TYPE_ZERO;
	block->permutation = xm_dim_identity_permutation(blkidx.n);
	block->scalar = 1.0;
	block->data_ptr = XM_NULL_PTR;
}

void
xm_tensor_set_canonical_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    uintptr_t data_ptr)
{
	struct xm_block *block;

	assert(data_ptr != XM_NULL_PTR);

	block = tensor_get_block(tensor, blkidx);
	block->type = XM_BLOCK_TYPE_CANONICAL;
	block->permutation = xm_dim_identity_permutation(blkidx.n);
	block->scalar = 1.0;
	block->data_ptr = data_ptr;
}

void
xm_tensor_set_derivative_block(xm_tensor_t *tensor, xm_dim_t blkidx,
    xm_dim_t source_blkidx, xm_dim_t permutation, xm_scalar_t scalar)
{
	struct xm_block *block;
	xm_dim_t blkdims1, blkdims2, nblocks;
	int type;

	type = xm_tensor_get_block_type(tensor, source_blkidx);
	if (type != XM_BLOCK_TYPE_CANONICAL)
		fatal("derivative blocks must have canonical source blocks");
	blkdims1 = xm_block_space_get_block_dims(tensor->bs, blkidx);
	blkdims1 = xm_dim_permute(&blkdims1, &permutation);
	blkdims2 = xm_block_space_get_block_dims(tensor->bs, source_blkidx);
	if (xm_dim_ne(&blkdims1, &blkdims2))
		fatal("invalid block permutation");

	nblocks = xm_tensor_get_nblocks(tensor);
	block = tensor_get_block(tensor, blkidx);
	block->type = XM_BLOCK_TYPE_DERIVATIVE;
	block->permutation = permutation;
	block->scalar = scalar;
	block->data_ptr = (uintptr_t)xm_dim_offset(&source_blkidx, &nblocks);
}

uintptr_t
xm_tensor_allocate_block_data(xm_tensor_t *tensor, xm_dim_t blkidx)
{
	xm_dim_t blkdims;
	size_t size;

	blkdims = xm_tensor_get_block_dims(tensor, blkidx);
	size = xm_dim_dot(&blkdims) * sizeof(xm_scalar_t);

	return xm_allocator_allocate(tensor->allocator, size);
}

uintptr_t
xm_tensor_get_block_data_ptr(const xm_tensor_t *tensor, xm_dim_t blkidx)
{
	struct xm_block *block;

	block = tensor_get_block(tensor, blkidx);
	if (block->type == XM_BLOCK_TYPE_CANONICAL)
		return block->data_ptr;
	if (block->type == XM_BLOCK_TYPE_DERIVATIVE)
		return tensor->blocks[block->data_ptr].data_ptr;
	return XM_NULL_PTR;
}

void
xm_tensor_free_block_data(xm_tensor_t *tensor)
{
	xm_dim_t idx, nblocks;
	uintptr_t data_ptr;
	int type;

	nblocks = xm_tensor_get_nblocks(tensor);
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		type = xm_tensor_get_block_type(tensor, idx);
		if (type == XM_BLOCK_TYPE_CANONICAL) {
			data_ptr = xm_tensor_get_block_data_ptr(tensor, idx);
			xm_allocator_deallocate(tensor->allocator, data_ptr);
		}
		xm_tensor_set_zero_block(tensor, idx);
		xm_dim_inc(&idx, &nblocks);
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

void
xm_tensor_unfold_block(xm_tensor_t *tensor, xm_dim_t blkidx, xm_dim_t mask_i,
    xm_dim_t mask_j, const xm_scalar_t *from, xm_scalar_t *to, size_t stride)
{
	struct xm_block *block = tensor_get_block(tensor, blkidx);
	xm_dim_t blkdims, blkdimsp, elidx, idx, permutation;
	size_t ii, jj, kk, offset, inc, lead_ii, lead_ii_nel;
	size_t block_size_i, block_size_j;
	xm_scalar_t scalar = block->scalar;

	assert(from);
	assert(to);

	blkdims = xm_tensor_get_block_dims(tensor, blkidx);
	block_size_i = xm_dim_dot_mask(&blkdims, &mask_i);
	block_size_j = xm_dim_dot_mask(&blkdims, &mask_j);
	permutation = block->permutation;
	blkdimsp = xm_dim_permute(&blkdims, &permutation);
	elidx = xm_dim_zero(blkdims.n);

	inc = 1;
	lead_ii_nel = 1;

	if (mask_i.n > 0) {
		lead_ii = mask_i.i[0];
		for (kk = 0; kk < permutation.i[lead_ii]; kk++)
			inc *= blkdimsp.i[kk];
		for (ii = 0; ii < mask_i.n-1; ii++)
			mask_i.i[ii] = mask_i.i[ii+1];
		mask_i.n--;
		lead_ii_nel = blkdims.i[lead_ii];
	}
	if (inc == 1) {
		for (jj = 0; jj < block_size_j; jj++) {
			xm_dim_zero_mask(&elidx, &mask_i);
			for (ii = 0; ii < block_size_i;
			    ii += lead_ii_nel) {
				idx = xm_dim_permute(&elidx, &permutation);
				offset = xm_dim_offset(&idx, &blkdimsp);
				memcpy(&to[jj * stride + ii],
				    from + offset,
				    sizeof(xm_scalar_t) * lead_ii_nel);
				xm_dim_inc_mask(&elidx, &blkdims, &mask_i);
			}
			xm_dim_inc_mask(&elidx, &blkdims, &mask_j);
		}
	} else {
		for (jj = 0; jj < block_size_j; jj++) {
			xm_dim_zero_mask(&elidx, &mask_i);
			for (ii = 0; ii < block_size_i;
			    ii += lead_ii_nel) {
				idx = xm_dim_permute(&elidx, &permutation);
				offset = xm_dim_offset(&idx, &blkdimsp);
				for (kk = 0; kk < lead_ii_nel; kk++) {
					to[jj * stride + ii + kk] =
					    from[offset];
					offset += inc;
				}
				xm_dim_inc_mask(&elidx, &blkdims, &mask_i);
			}
			xm_dim_inc_mask(&elidx, &blkdims, &mask_j);
		}
	}
	for (jj = 0; jj < block_size_j; jj++)
		for (ii = 0; ii < block_size_i; ii++)
			to[jj * stride + ii] *= scalar;
}

void
xm_tensor_fold_block(xm_tensor_t *tensor, xm_dim_t blkidx, xm_dim_t mask_i,
    xm_dim_t mask_j, const xm_scalar_t *from, xm_scalar_t *to, size_t stride)
{
	struct xm_block *block = tensor_get_block(tensor, blkidx);
	xm_dim_t blkdims, elidx;
	size_t ii, jj, offset, block_size_i, block_size_j;

	assert(from);
	assert(to);
	assert(block->type == XM_BLOCK_TYPE_CANONICAL);
	assert(block->data_ptr != XM_NULL_PTR);

	blkdims = xm_tensor_get_block_dims(tensor, blkidx);
	block_size_i = xm_dim_dot_mask(&blkdims, &mask_i);
	block_size_j = xm_dim_dot_mask(&blkdims, &mask_j);
	elidx = xm_dim_zero(blkdims.n);

	assert(mask_i.n > 0);/*XXX*/
	assert(mask_i.i[0] == 0);
	for (ii = 0; ii < mask_i.n-1; ii++)
		mask_i.i[ii] = mask_i.i[ii+1];
	mask_i.n--;

	for (jj = 0; jj < block_size_j; jj++) {
		xm_dim_zero_mask(&elidx, &mask_i);
		for (ii = 0; ii < block_size_i; ii += blkdims.i[0]) {
			offset = xm_dim_offset(&elidx, &blkdims);
			memcpy(to + offset,
			    &from[jj * stride + ii],
			    sizeof(xm_scalar_t) * blkdims.i[0]);
			xm_dim_inc_mask(&elidx, &blkdims, &mask_i);
		}
		xm_dim_inc_mask(&elidx, &blkdims, &mask_j);
	}
}
