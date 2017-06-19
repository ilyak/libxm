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

#include <stdlib.h>

#include "xm.h"
#include "util.h"

void
xm_set(xm_tensor_t *a, xm_scalar_t x)
{
	xm_allocator_t *alloc;
	const xm_block_space_t *bs;
	xm_dim_t nblocks;
	size_t i, blockcount, maxblksize;
	xm_scalar_t *buf;

	alloc = xm_tensor_get_allocator(a);
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
	size_t blksize;
	uintptr_t data_ptr;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < blockcount; i++) {
		xm_dim_t idx = xm_dim_from_offset(i, &nblocks);
		int type = xm_tensor_get_block_type(a, idx);
		if (type == XM_BLOCK_TYPE_CANONICAL) {
			blksize = xm_tensor_get_block_size(a, idx);
			data_ptr = xm_tensor_get_block_data_ptr(a, idx);
			xm_allocator_write(alloc, data_ptr, buf,
			    blksize * sizeof(xm_scalar_t));
		}
	}
}
	free(buf);
}
