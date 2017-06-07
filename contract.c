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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(XM_SCALAR_FLOAT)
#define dgemm_ sgemm_
#elif defined(XM_SCALAR_DOUBLE_COMPLEX)
#define dgemm_ zgemm_
#elif defined(XM_SCALAR_FLOAT_COMPLEX)
#define dgemm_ cgemm_
#endif

void dgemm_(char *, char *, long int *, long int *, long int *, xm_scalar_t *,
    xm_scalar_t *, long int *, xm_scalar_t *, long int *, xm_scalar_t *,
    xm_scalar_t *, long int *);

static void
xm_dgemm(char transa, char transb, long int m, long int n, long int k,
    xm_scalar_t alpha, xm_scalar_t *a, long int lda, xm_scalar_t *b,
    long int ldb, xm_scalar_t beta, xm_scalar_t *c, long int ldc)
{
	dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb,
	    &beta, c, &ldc);
}

static void
fatal(const char *msg)
{
	fprintf(stderr, "libxm: %s\n", msg);
	abort();
}

static void
make_masks(const char *str1, const char *str2, xm_dim_t *mask1, xm_dim_t *mask2)
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

static xm_dim_t *
get_canonical_block_list(xm_tensor_t *tensor, size_t *ncanblksout)
{
	xm_dim_t idx, nblocks, *canblks = NULL;
	size_t ncanblks = 0;
	int type;

	nblocks = xm_tensor_get_nblocks(tensor);
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		type = xm_tensor_get_block_type(tensor, idx);
		if (type == XM_BLOCK_TYPE_CANONICAL) {
			ncanblks++;
			canblks = realloc(canblks, ncanblks * sizeof *canblks);
			if (canblks == NULL)
				fatal("out of memory");
			canblks[ncanblks-1] = idx;
		}
		xm_dim_inc(&idx, &nblocks);
	}
	*ncanblksout = ncanblks;
	return canblks;
}

static void
compute_block(xm_scalar_t alpha, xm_tensor_t *a, xm_tensor_t *b,
    xm_scalar_t beta, xm_tensor_t *c, xm_dim_t cidxa, xm_dim_t aidxa,
    xm_dim_t cidxb, xm_dim_t aidxb, xm_dim_t cidxc, xm_dim_t aidxc,
    xm_dim_t blkidxc, xm_scalar_t *buf)
{
	xm_allocator_t *alloca = xm_tensor_get_allocator(a);
	xm_allocator_t *allocb = xm_tensor_get_allocator(b);
	xm_allocator_t *allocc = xm_tensor_get_allocator(c);
	const xm_block_space_t *bsa = xm_tensor_get_block_space(a);
	const xm_block_space_t *bsb = xm_tensor_get_block_space(b);
	const xm_block_space_t *bsc = xm_tensor_get_block_space(c);
	size_t maxblocksizea = xm_block_space_get_largest_block_size(bsa);
	size_t maxblocksizeb = xm_block_space_get_largest_block_size(bsb);
	size_t maxblocksizec = xm_block_space_get_largest_block_size(bsc);
	xm_dim_t dims, blkidxa, blkidxb, nblocksa, nblocksb;
	xm_scalar_t *bufa1, *bufa2, *bufb1, *bufb2, *bufc1, *bufc2;
	size_t ii, jj, m, n, k, nblk_k, size;
	uintptr_t data_ptr;

	bufa1 = buf;
	bufa2 = bufa1 + maxblocksizea;
	bufb1 = bufa2 + maxblocksizea;
	bufb2 = bufb1 + maxblocksizeb;
	bufc1 = bufb2 + maxblocksizeb;
	bufc2 = bufc1 + maxblocksizec;

	blkidxa = xm_dim_zero(xm_block_space_get_ndims(bsa));
	blkidxb = xm_dim_zero(xm_block_space_get_ndims(bsb));
	xm_dim_set_mask(&blkidxa, &aidxa, &blkidxc, &cidxc);
	xm_dim_set_mask(&blkidxb, &aidxb, &blkidxc, &aidxc);
	nblocksa = xm_tensor_get_nblocks(a);
	nblocksb = xm_tensor_get_nblocks(b);
	nblk_k = xm_dim_dot_mask(&nblocksa, &cidxa);

	dims = xm_tensor_get_block_dims(c, blkidxc);
	m = xm_dim_dot_mask(&dims, &cidxc);
	n = xm_dim_dot_mask(&dims, &aidxc);
	size = xm_dim_dot(&dims);
	data_ptr = xm_tensor_get_block_data_ptr(c, blkidxc);
	xm_allocator_read(allocc, data_ptr, bufc2, size * sizeof(xm_scalar_t));
	if (aidxc.n > 0 && aidxc.i[0] == 0) {
		xm_tensor_unfold_block(c, blkidxc, aidxc, cidxc,
		    bufc2, bufc1, n);
		for (jj = 0; jj < m; jj++)
			for (ii = 0; ii < n; ii++)
				bufc1[jj * n + ii] *= beta;
	} else {
		xm_tensor_unfold_block(c, blkidxc, cidxc, aidxc,
		    bufc2, bufc1, m);
		for (jj = 0; jj < n; jj++)
			for (ii = 0; ii < m; ii++)
				bufc1[jj * m + ii] *= beta;
	}
	if (alpha == 0.0)
		goto done;
	for (ii = 0; ii < nblk_k; ii++) {
		int blktypea = xm_tensor_get_block_type(a, blkidxa);
		int blktypeb = xm_tensor_get_block_type(b, blkidxb);
		if (blktypea != XM_BLOCK_TYPE_ZERO &&
		    blktypeb != XM_BLOCK_TYPE_ZERO) {
			dims = xm_tensor_get_block_dims(a, blkidxa);
			k = xm_dim_dot_mask(&dims, &cidxa);
			size = xm_dim_dot(&dims);
			data_ptr = xm_tensor_get_block_data_ptr(a, blkidxa);
			xm_allocator_read(alloca, data_ptr, bufa1,
			    size * sizeof(xm_scalar_t));
			xm_tensor_unfold_block(a, blkidxa, cidxa,
			    aidxa, bufa1, bufa2, k);

			dims = xm_tensor_get_block_dims(b, blkidxb);
			size = xm_dim_dot(&dims);
			data_ptr = xm_tensor_get_block_data_ptr(b, blkidxb);
			xm_allocator_read(allocb, data_ptr, bufb1,
			    size * sizeof(xm_scalar_t));
			xm_tensor_unfold_block(b, blkidxb, cidxb,
			    aidxb, bufb1, bufb2, k);

			if (aidxc.n > 0 && aidxc.i[0] == 0) {
				xm_dgemm('T', 'N', (int)n, (int)m, (int)k,
				    alpha, bufb2, (int)k, bufa2, (int)k, 1.0,
				    bufc1, (int)n);
			} else {
				xm_dgemm('T', 'N', (int)m, (int)n, (int)k,
				    alpha, bufa2, (int)k, bufb2, (int)k, 1.0,
				    bufc1, (int)m);
			}
		}
		xm_dim_inc_mask(&blkidxa, &nblocksa, &cidxa);
		xm_dim_inc_mask(&blkidxb, &nblocksb, &cidxb);
	}
done:
	if (aidxc.n > 0 && aidxc.i[0] == 0) {
		xm_tensor_fold_block(c, blkidxc, aidxc, cidxc,
		    bufc1, bufc2, n);
	} else {
		xm_tensor_fold_block(c, blkidxc, cidxc, aidxc,
		    bufc1, bufc2, m);
	}
	dims = xm_tensor_get_block_dims(c, blkidxc);
	size = xm_dim_dot(&dims);
	data_ptr = xm_tensor_get_block_data_ptr(c, blkidxc);
	xm_allocator_write(allocc, data_ptr, bufc2, size * sizeof(xm_scalar_t));
}

void
xm_contract(xm_scalar_t alpha, xm_tensor_t *a, xm_tensor_t *b,
    xm_scalar_t beta, xm_tensor_t *c, const char *idxa, const char *idxb,
    const char *idxc)
{
	const xm_block_space_t *bsa, *bsb, *bsc;
	xm_dim_t cidxa, aidxa, cidxb, aidxb, cidxc, aidxc, *canblks;
	size_t i, size, ncanblks, maxblocksizea, maxblocksizeb, maxblocksizec;

	bsa = xm_tensor_get_block_space(a);
	bsb = xm_tensor_get_block_space(b);
	bsc = xm_tensor_get_block_space(c);

	if (strlen(idxa) != xm_block_space_get_ndims(bsa))
		fatal("bad contraction indices");
	if (strlen(idxb) != xm_block_space_get_ndims(bsb))
		fatal("bad contraction indices");
	if (strlen(idxc) != xm_block_space_get_ndims(bsc))
		fatal("bad contraction indices");

	make_masks(idxa, idxb, &cidxa, &cidxb);
	make_masks(idxc, idxa, &cidxc, &aidxa);
	make_masks(idxc, idxb, &aidxc, &aidxb);

	if (aidxa.n + cidxa.n != xm_block_space_get_ndims(bsa))
		fatal("bad contraction indices");
	if (aidxb.n + cidxb.n != xm_block_space_get_ndims(bsb))
		fatal("bad contraction indices");
	if (aidxc.n + cidxc.n != xm_block_space_get_ndims(bsc))
		fatal("bad contraction indices");
	if (!(aidxc.n > 0 && aidxc.i[0] == 0) &&
	    !(cidxc.n > 0 && cidxc.i[0] == 0))
		fatal("bad contraction indices");

	for (i = 0; i < cidxa.n; i++)
		if (!xm_block_space_eq1(bsa, cidxa.i[i], bsb, cidxb.i[i]))
			fatal("inconsistent a and b tensor block spaces");
	for (i = 0; i < cidxc.n; i++)
		if (!xm_block_space_eq1(bsc, cidxc.i[i], bsa, aidxa.i[i]))
			fatal("inconsistent a and c tensor block spaces");
	for (i = 0; i < aidxc.n; i++)
		if (!xm_block_space_eq1(bsc, aidxc.i[i], bsb, aidxb.i[i]))
			fatal("inconsistent b and c tensor block spaces");

	maxblocksizea = xm_block_space_get_largest_block_size(bsa);
	maxblocksizeb = xm_block_space_get_largest_block_size(bsb);
	maxblocksizec = xm_block_space_get_largest_block_size(bsc);
	canblks = get_canonical_block_list(c, &ncanblks);
	size = 2 * (maxblocksizea + maxblocksizeb + maxblocksizec);
#ifdef _OPENMP
#pragma omp parallel private(i)
#endif
{
	xm_scalar_t *buf = malloc(size * sizeof(xm_scalar_t));
	if (buf == NULL)
		fatal("out of memory");
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (i = 0; i < ncanblks; i++) {
		compute_block(alpha, a, b, beta, c, cidxa, aidxa, cidxb,
		    aidxb, cidxc, aidxc, canblks[i], buf);
	}
	free(buf);
}
	free(canblks);
}
