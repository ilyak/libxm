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
#include <string.h>

#include "xm.h"
#include "util.h"

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
		fatal("out of memory");
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
xm_copy(xm_tensor_t *a, xm_scalar_t s, const xm_tensor_t *b, const char *idxa,
    const char *idxb)
{
	xm_add(0, a, s, b, idxa, idxb);
}

void
xm_add(xm_scalar_t alpha, xm_tensor_t *a, xm_scalar_t beta,
    const xm_tensor_t *b, const char *idxa, const char *idxb)
{
	const xm_block_space_t *bsa, *bsb;
	xm_dim_t cidxa, cidxb, zero, nblocksa;
	size_t i, blockcount, maxblksize;

	bsa = xm_tensor_get_block_space(a);
	bsb = xm_tensor_get_block_space(b);
	if (strlen(idxa) != xm_block_space_get_ndims(bsa))
		fatal("idxa does not match tensor dimensions");
	if (strlen(idxb) != xm_block_space_get_ndims(bsb))
		fatal("idxb does not match tensor dimensions");
	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	if (cidxa.n != xm_block_space_get_ndims(bsa) ||
	    cidxb.n != xm_block_space_get_ndims(bsb))
		fatal("index spaces do not match");
	for (i = 0; i < cidxa.n; i++)
		if (!xm_block_space_eq1(bsa, cidxa.i[i], bsb, cidxb.i[i]))
			fatal("inconsistent block-spaces");

	zero = xm_dim_zero(0);
	maxblksize = xm_block_space_get_largest_block_size(bsa);
	nblocksa = xm_tensor_get_nblocks(a);
	blockcount = xm_dim_dot(&nblocksa);
#ifdef _OPENMP
#pragma omp parallel private(i)
#endif
{
	xm_dim_t ia, ib;
	xm_scalar_t *buf1, *buf2;
	size_t j, blksize;
	int typea, typeb;

	if ((buf1 = malloc(2 * maxblksize * sizeof *buf1)) == NULL)
		fatal("out of memory");
	buf2 = buf1 + maxblksize;
	ib = xm_dim_zero(cidxb.n);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < blockcount; i++) {
		ia = xm_dim_from_offset(i, &nblocksa);
		typea = xm_tensor_get_block_type(a, ia);
		if (typea == XM_BLOCK_TYPE_CANONICAL) {
			xm_dim_set_mask(&ib, &cidxb, &ia, &cidxa);
			typeb = xm_tensor_get_block_type(b, ib);
			blksize = xm_tensor_get_block_size(b, ib);
			if (beta == 0 || typeb == XM_BLOCK_TYPE_ZERO) {
				memset(buf2, 0, blksize * sizeof *buf2);
			} else {
				xm_scalar_t scalar = beta;
				scalar *= xm_tensor_get_block_scalar(b, ib);
				xm_tensor_read_block(b, ib, buf2);
				xm_tensor_unfold_block(b, ib, cidxb, zero, buf2,
				    buf1, blksize);
				for (j = 0; j < blksize; j++)
					buf1[j] *= scalar;
				xm_tensor_fold_block(a, ia, cidxa, zero, buf1,
				    buf2, blksize);
			}
			if (alpha == 0)
				xm_tensor_write_block(a, ia, buf2);
			else {
				xm_tensor_read_block(a, ia, buf1);
				for (j = 0; j < blksize; j++)
					buf1[j] = alpha * buf1[j] + buf2[j];
				xm_tensor_write_block(a, ia, buf1);
			}
		}
	}
	free(buf1);
}
}

void
xm_print_banner(void)
{
	printf("Libxm Tensor Library\n");
	printf("Copyright (c) 2014-2017 Ilya Kaliman\n");
	printf("https://github.com/ilyak/libxm\n");
}
