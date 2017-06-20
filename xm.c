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

#include <stdio.h>
#include <stdlib.h>

#include "xm.h"
#include "util.h"

void
xm_copy(xm_tensor_t *a, const xm_tensor_t *b, xm_scalar_t s)
{
	const xm_block_space_t *bsa, *bsb;
	xm_dim_t nblocks;
	size_t blockcount, maxblksize;

	bsa = xm_tensor_get_block_space(a);
	bsb = xm_tensor_get_block_space(b);
	if (!xm_block_space_eq(bsa, bsb))
		xm_fatal("%s: block spaces do not match", __func__);
	maxblksize = xm_block_space_get_largest_block_size(bsa);
	nblocks = xm_tensor_get_nblocks(a);
	blockcount = xm_dim_dot(&nblocks);
#ifdef _OPENMP
#pragma omp parallel
#endif
{
	xm_scalar_t *buf;
	size_t i, j, blksize;

	if ((buf = malloc(maxblksize * sizeof *buf)) == NULL)
		xm_fatal("%s: out of memory", __func__);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < blockcount; i++) {
		xm_dim_t idx = xm_dim_from_offset(i, &nblocks);
		int typea = xm_tensor_get_block_type(a, idx);
		int typeb = xm_tensor_get_block_type(b, idx);
		if (typea != typeb)
			xm_fatal("%s: block structures do not match", __func__);
		if (typea == XM_BLOCK_TYPE_CANONICAL) {
			xm_tensor_read_block(b, idx, buf);
			blksize = xm_tensor_get_block_size(b, idx);
			for (j = 0; j < blksize; j++)
				buf[j] *= s;
			xm_tensor_write_block(a, idx, buf);
		}
	}
	free(buf);
}
}

void
xm_set(xm_tensor_t *a, xm_scalar_t x)
{
	const xm_block_space_t *bs;
	xm_dim_t nblocks;
	size_t i, blockcount, maxblksize;
	xm_scalar_t *buf;

	bs = xm_tensor_get_block_space(a);
	maxblksize = xm_block_space_get_largest_block_size(bs);
	if ((buf = malloc(maxblksize * sizeof *buf)) == NULL)
		xm_fatal("%s: out of memory", __func__);
	for (i = 0; i < maxblksize; i++)
		buf[i] = x;
	nblocks = xm_tensor_get_nblocks(a);
	blockcount = xm_dim_dot(&nblocks);
#ifdef _OPENMP
#pragma omp parallel private(i)
#endif
{
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < blockcount; i++) {
		xm_dim_t idx = xm_dim_from_offset(i, &nblocks);
		int type = xm_tensor_get_block_type(a, idx);
		if (type == XM_BLOCK_TYPE_CANONICAL)
			xm_tensor_write_block(a, idx, buf);
	}
}
	free(buf);
}

void
xm_print_banner(void)
{
	printf("libxm (c) 2014-2017 Ilya Kaliman\n");
	printf("Efficient operations on block tensors\n");
	printf("https://github.com/ilyak/libxm\n");
}
