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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static void
fatal(const char *fmt, ...)
{
	va_list ap;

	fprintf(stderr, "libxm: ");
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fprintf(stderr, "\n");
	abort();
}

void
xm_copy(xm_tensor_t *a, const xm_tensor_t *b, xm_scalar_t s)
{
	xm_allocator_t *alloca, *allocb;
	const xm_block_space_t *bsa, *bsb;
	xm_dim_t nblocks;
	size_t blockcount, maxblksize;

	bsa = xm_tensor_get_block_space(a);
	bsb = xm_tensor_get_block_space(b);
	if (!xm_block_space_eq(bsa, bsb))
		fatal("%s: block spaces do not match", __func__);
	alloca = xm_tensor_get_allocator(a);
	allocb = xm_tensor_get_allocator(b);
	maxblksize = xm_block_space_get_largest_block_size(bsa);
	nblocks = xm_tensor_get_nblocks(a);
	blockcount = xm_dim_dot(&nblocks);
#ifdef _OPENMP
#pragma omp parallel
#endif
{
	xm_scalar_t *buf;
	size_t i, j, blksize;
	uintptr_t data_ptr;

	if ((buf = malloc(maxblksize * sizeof *buf)) == NULL)
		fatal("%s: out of memory", __func__);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < blockcount; i++) {
		xm_dim_t idx = xm_dim_from_offset(i, &nblocks);
		int typea = xm_tensor_get_block_type(a, idx);
		int typeb = xm_tensor_get_block_type(b, idx);
		if (typea != typeb)
			fatal("%s: block structures do not match", __func__);
		if (typea == XM_BLOCK_TYPE_CANONICAL) {
			blksize = xm_tensor_get_block_size(a, idx);
			data_ptr = xm_tensor_get_block_data_ptr(b, idx);
			xm_allocator_read(allocb, data_ptr, buf,
			    blksize * sizeof(xm_scalar_t));
			for (j = 0; j < blksize; j++)
				buf[j] *= s;
			data_ptr = xm_tensor_get_block_data_ptr(a, idx);
			xm_allocator_write(alloca, data_ptr, buf,
			    blksize * sizeof(xm_scalar_t));
		}
	}
	free(buf);
}
}
