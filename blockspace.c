/*
 * Copyright (c) 2017 Ilya Kaliman
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
#include <stdlib.h>
#include <string.h>

struct xm_block_space {
	xm_dim_t dims, nblocks;
	size_t *splits[XM_MAX_DIM];
};

xm_block_space_t *
xm_block_space_create(const xm_dim_t *dims)
{
	xm_block_space_t *ret;
	size_t i;

	assert(dims->n > 0 && dims->n <= XM_MAX_DIM);

	if ((ret = calloc(1, sizeof *ret)) == NULL)
		return NULL;
	ret->dims = *dims;
	ret->nblocks = xm_dim_same(dims->n, 1);
	for (i = 0; i < dims->n; i++) {
		if ((ret->splits[i] = malloc(2 * sizeof(size_t))) == NULL) {
			xm_block_space_free(ret);
			return NULL;
		}
		ret->splits[i][0] = 0;
		ret->splits[i][1] = dims->i[i];
	}
	return ret;
}

size_t
xm_block_space_get_ndims(const xm_block_space_t *bs)
{
	return bs->dims.n;
}

xm_dim_t
xm_block_space_get_abs_dims(const xm_block_space_t *bs)
{
	return bs->dims;
}

xm_dim_t
xm_block_space_get_nblocks(const xm_block_space_t *bs)
{
	return bs->nblocks;
}

void
xm_block_space_split(xm_block_space_t *bs, size_t dim, size_t x)
{
	size_t i, *nptr;

	assert(dim < bs->dims.n);
	assert(x < bs->dims.i[dim]);

	for (i = 0; i < bs->nblocks.i[dim]+1; i++) {
		if (bs->splits[dim][i] > x) {
			if ((nptr = realloc(bs->splits[dim],
			    (bs->nblocks.i[dim]+2)*sizeof(size_t))) == NULL)
				return;
			bs->splits[dim] = nptr;
			break;
		} else if (bs->splits[dim][i] == x)
			return;
	}
	memmove(bs->splits[dim]+i+1, bs->splits[dim]+i,
	    (bs->nblocks.i[dim]-i+1)*sizeof(size_t));
	bs->splits[dim][i] = x;
	bs->nblocks.i[dim]++;
}

xm_dim_t
xm_block_space_get_block_dims(const xm_block_space_t *bs,
    const xm_dim_t *blkidx)
{
	xm_dim_t blkdims;
	size_t i;

	assert(bs->dims.n == blkidx->n);
	assert(xm_dim_less(blkidx, &bs->nblocks));

	blkdims.n = bs->dims.n;
	for (i = 0; i < blkdims.n; i++) {
		blkdims.i[i] = bs->splits[i][blkidx->i[i]+1] -
		    bs->splits[i][blkidx->i[i]];
	}
	return blkdims;
}

int
xm_block_space_eq(const xm_block_space_t *a, const xm_block_space_t *b)
{
	size_t i;

	if (a->dims.n != b->dims.n)
		return 0;
	for (i = 0; i < a->dims.n; i++)
		if (!xm_block_space_eq1(a, i, b, i))
			return 0;
	return 1;
}

int
xm_block_space_eq1(const xm_block_space_t *a, size_t dima,
    const xm_block_space_t *b, size_t dimb)
{
	size_t i;

	assert(dima < a->nblocks.n);
	assert(dimb < b->nblocks.n);

	if (a->nblocks.i[dima] != b->nblocks.i[dimb])
		return 0;
	for (i = 0; i < a->nblocks.i[dima]+1; i++)
		if (a->splits[dima][i] != b->splits[dimb][i])
			return 0;
	return 1;
}

void
xm_block_space_free(xm_block_space_t *bs)
{
	size_t i;

	if (bs) {
		for (i = 0; i < bs->dims.n; i++)
			free(bs->splits[i]);
		free(bs);
	}
}
