/*
 * Copyright (c) 2014-2018 Ilya Kaliman
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
#include <string.h>

#include "xm.h"
#include "util.h"

struct blockpair {
	xm_dim_t blkidxa, blkidxb;
	xm_dim_t *blkidxc;
	xm_scalar_t *alpha;
	size_t nblkidxc;
};

void sgemm_(char *, char *, long int *, long int *, long int *,
    float *, float *, long int *,
    float *, long int *, float *,
    float *, long int *);
void cgemm_(char *, char *, long int *, long int *, long int *,
    float complex *, float complex *, long int *,
    float complex *, long int *, float complex *,
    float complex *, long int *);
void dgemm_(char *, char *, long int *, long int *, long int *,
    double *, double *, long int *,
    double *, long int *, double *,
    double *, long int *);
void zgemm_(char *, char *, long int *, long int *, long int *,
    double complex *, double complex *, long int *,
    double complex *, long int *, double complex *,
    double complex *, long int *);

static void
xgemm(char transa, char transb, long int m, long int n, long int k,
    xm_scalar_t alpha, void *a, long int lda, void *b, long int ldb,
    xm_scalar_t beta, void *c, long int ldc, int type)
{
	switch (type) {
	case XM_SCALAR_FLOAT: {
		float al = alpha, bt = beta;
		sgemm_(&transa, &transb, &m, &n, &k, &al, a, &lda, b, &ldb,
		    &bt, c, &ldc);
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex al = alpha, bt = beta;
		cgemm_(&transa, &transb, &m, &n, &k, &al, a, &lda, b, &ldb,
		    &bt, c, &ldc);
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double al = alpha, bt = beta;
		dgemm_(&transa, &transb, &m, &n, &k, &al, a, &lda, b, &ldb,
		    &bt, c, &ldc);
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex al = alpha, bt = beta;
		zgemm_(&transa, &transb, &m, &n, &k, &al, a, &lda, b, &ldb,
		    &bt, c, &ldc);
		return;
	}
	}
}

static int
same_contraction(const struct blockpair *pairs, size_t i, size_t j,
    xm_dim_t aidxa, xm_dim_t aidxb, const xm_tensor_t *a, const xm_tensor_t *b)
{
	xm_dim_t dia, dja, dib, djb, pia, pja, pib, pjb;
	size_t ii;

	dia = pairs[i].blkidxa;
	dja = pairs[j].blkidxa;
	dib = pairs[i].blkidxb;
	djb = pairs[j].blkidxb;
	if (xm_tensor_get_block_data_ptr(a, dia) !=
	    xm_tensor_get_block_data_ptr(a, dja) ||
	    xm_tensor_get_block_data_ptr(b, dib) !=
	    xm_tensor_get_block_data_ptr(b, djb))
		return 0;
	pia = xm_tensor_get_block_permutation(a, dia);
	pja = xm_tensor_get_block_permutation(a, dja);
	pib = xm_tensor_get_block_permutation(b, dib);
	pjb = xm_tensor_get_block_permutation(b, djb);
	for (ii = 0; ii < aidxa.n; ii++) {
		if (pia.i[aidxa.i[ii]] != pja.i[aidxa.i[ii]])
			return 0;
	}
	for (ii = 0; ii < aidxb.n; ii++) {
		if (pib.i[aidxb.i[ii]] != pjb.i[aidxb.i[ii]])
			return 0;
	}
	return 1;
}

static void
make_pairs(const xm_tensor_t *a, const xm_tensor_t *b, xm_dim_t cidxa,
    xm_dim_t aidxa, xm_dim_t cidxb, xm_dim_t aidxb, xm_dim_t cidxc,
    xm_dim_t aidxc, const xm_dim_t *canblkc, size_t ncanblkc,
    struct blockpair **pairs, size_t *npairs)
{
	struct blockpair *p = NULL;
	xm_dim_t nblocksa, nblocksb, blkidxa, blkidxb;
	size_t i, j, k, nblkk, np = 0;

	nblocksa = xm_tensor_get_nblocks(a);
	nblocksb = xm_tensor_get_nblocks(b);
	nblkk = xm_dim_dot_mask(&nblocksa, &cidxa);

	for (i = 0; i < ncanblkc; i++) {
		xm_dim_t blkidxc = canblkc[i];
		blkidxa = xm_dim_zero(nblocksa.n);
		blkidxb = xm_dim_zero(nblocksb.n);
		xm_dim_set_mask(&blkidxa, &aidxa, &blkidxc, &cidxc);
		xm_dim_set_mask(&blkidxb, &aidxb, &blkidxc, &aidxc);
		for (j = 0; j < nblkk; j++) {
			int blktypea = xm_tensor_get_block_type(a, blkidxa);
			int blktypeb = xm_tensor_get_block_type(b, blkidxb);
			if (blktypea != XM_BLOCK_TYPE_ZERO &&
			    blktypeb != XM_BLOCK_TYPE_ZERO) {
				np++;
				p = realloc(p, np * sizeof(*p));
				if (p == NULL)
					fatal("out of memory");
				p[np-1].alpha = malloc(sizeof(xm_scalar_t));
				if (p[np-1].alpha == NULL)
					fatal("out of memory");
				p[np-1].alpha[0] = 1;
				p[np-1].blkidxa = blkidxa;
				p[np-1].blkidxb = blkidxb;
				p[np-1].nblkidxc = 1;
				p[np-1].blkidxc = malloc(sizeof(xm_dim_t));
				if (p[np-1].blkidxc == NULL)
					fatal("out of memory");
				p[np-1].blkidxc[0] = blkidxc;
			}
			xm_dim_inc_mask(&blkidxa, &nblocksa, &cidxa);
			xm_dim_inc_mask(&blkidxb, &nblocksb, &cidxb);
		}
	}
	for (i = 0; i < np; i++) {
		for (j = i+1; j < np; j++) {
			if (same_contraction(p, i, j, aidxa, aidxb, a, b)) {
				for (k = 0; k < p[i].nblkidxc; k++) {
					if (xm_dim_eq(&p[i].blkidxc[k],
						      &p[j].blkidxc[0])) {
						p[i].alpha[k] += 1;
						goto next;
					}
				}
				p[i].nblkidxc++;
				p[i].blkidxc = realloc(p[i].blkidxc,
				    p[i].nblkidxc * sizeof(xm_dim_t));
				if (p[i].blkidxc == NULL)
					fatal("out of memory");
				p[i].blkidxc[p[i].nblkidxc-1] = p[j].blkidxc[0];

				p[i].alpha = realloc(p[i].alpha,
				    p[i].nblkidxc * sizeof(xm_scalar_t));
				if (p[i].alpha == NULL)
					fatal("out of memory");
				p[i].alpha[p[i].nblkidxc-1] = 1;
next:
				free(p[j].alpha);
				free(p[j].blkidxc);
				memmove(p+j, p+j+1, (np-j-1) * sizeof(*p));
				np--;
				j--;
			}
		}
	}
	*pairs = p;
	*npairs = np;
}

void
xm_contract(xm_scalar_t alpha, const xm_tensor_t *a, const xm_tensor_t *b,
    xm_scalar_t beta, xm_tensor_t *c, const char *idxa, const char *idxb,
    const char *idxc)
{
	const xm_block_space_t *bsa, *bsb, *bsc;
	struct blockpair *pairs;
	xm_dim_t cidxa, aidxa, cidxb, aidxb, cidxc, aidxc, *canblkc;
	size_t i, j, ncanblkc, npairs;
	int type;

	if (xm_tensor_get_allocator(a) != xm_tensor_get_allocator(c) ||
	    xm_tensor_get_allocator(b) != xm_tensor_get_allocator(c))
		fatal("tensors must use same allocator");
	if (xm_tensor_get_scalar_type(a) != xm_tensor_get_scalar_type(c) ||
	    xm_tensor_get_scalar_type(b) != xm_tensor_get_scalar_type(c))
		fatal("tensors must have same scalar type");

	type = xm_tensor_get_scalar_type(a);
	bsa = xm_tensor_get_block_space(a);
	bsb = xm_tensor_get_block_space(b);
	bsc = xm_tensor_get_block_space(c);

	if (strlen(idxa) != xm_block_space_get_ndims(bsa))
		fatal("bad contraction indices");
	if (strlen(idxb) != xm_block_space_get_ndims(bsb))
		fatal("bad contraction indices");
	if (strlen(idxc) != xm_block_space_get_ndims(bsc))
		fatal("bad contraction indices");

	xm_make_masks(idxa, idxb, &cidxa, &cidxb);
	xm_make_masks(idxc, idxa, &cidxc, &aidxa);
	xm_make_masks(idxc, idxb, &aidxc, &aidxb);

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
			fatal("inconsistent a and b tensor block-spaces");
	for (i = 0; i < cidxc.n; i++)
		if (!xm_block_space_eq1(bsc, cidxc.i[i], bsa, aidxa.i[i]))
			fatal("inconsistent a and c tensor block-spaces");
	for (i = 0; i < aidxc.n; i++)
		if (!xm_block_space_eq1(bsc, aidxc.i[i], bsb, aidxb.i[i]))
			fatal("inconsistent b and c tensor block-spaces");

	xm_tensor_get_canonical_block_list(c, &canblkc, &ncanblkc);
	make_pairs(a, b, cidxa, aidxa, cidxb, aidxb, cidxc, aidxc,
	    canblkc, ncanblkc, &pairs, &npairs);

	size_t maxblkbytesa = xm_tensor_get_largest_block_bytes(a);
	size_t maxblkbytesb = xm_tensor_get_largest_block_bytes(b);
	size_t maxblkbytesc = xm_tensor_get_largest_block_bytes(c);
	size_t bufbytes = 2*maxblkbytesa + 2*maxblkbytesb + 3*maxblkbytesc;
	void *bufa1 = malloc(bufbytes);
	if (bufa1 == NULL)
		fatal("out of memory");
	void *bufa2 = (char *)bufa1 + maxblkbytesa;
	void *bufb1 = (char *)bufa2 + maxblkbytesa;
	void *bufb2 = (char *)bufb1 + maxblkbytesb;
	void *bufc1 = (char *)bufb2 + maxblkbytesb;
	void *bufc2 = (char *)bufc1 + maxblkbytesc;
	void *bufc3 = (char *)bufc2 + maxblkbytesc;

	for (i = 0; i < ncanblkc; i++) {
		xm_dim_t blkidxc = canblkc[i];
		size_t blksize = xm_tensor_get_block_size(c, blkidxc);

		xm_tensor_read_block(c, blkidxc, bufc1);
		xm_scalar_mul(bufc1, beta, blksize, type);
		xm_tensor_write_block(c, blkidxc, bufc1);
	}
	if (alpha == 0)
		goto done;
	for (i = 0; i < npairs; i++) {
		xm_dim_t blkidxa = pairs[i].blkidxa;
		xm_dim_t blkidxb = pairs[i].blkidxb;
		xm_dim_t blkdimsa = xm_tensor_get_block_dims(a, blkidxa);
		xm_dim_t blkdimsb = xm_tensor_get_block_dims(b, blkidxb);
		size_t m = xm_dim_dot_mask(&blkdimsa, &aidxa);
		size_t n = xm_dim_dot_mask(&blkdimsb, &aidxb);
		size_t k = xm_dim_dot_mask(&blkdimsa, &cidxa);

		xm_tensor_read_block(a, blkidxa, bufa1);
		xm_tensor_unfold_block(a, blkidxa, cidxa, aidxa,
		    bufa1, bufa2, k);
		xm_tensor_read_block(b, blkidxb, bufb1);
		xm_tensor_unfold_block(b, blkidxb, cidxb, aidxb,
		    bufb1, bufb2, k);
		xm_scalar_t sa = xm_tensor_get_block_scalar(a, blkidxa);
		xm_scalar_t sb = xm_tensor_get_block_scalar(b, blkidxb);

		if (aidxc.n > 0 && aidxc.i[0] == 0) {
			xgemm('T', 'N', (int)n, (int)m, (int)k, alpha*sa*sb,
			    bufb2, (int)k, bufa2, (int)k, 0, bufc3, (int)n,
			    type);
		} else {
			xgemm('T', 'N', (int)m, (int)n, (int)k, alpha*sa*sb,
			    bufa2, (int)k, bufb2, (int)k, 0, bufc3, (int)m,
			    type);
		}
		for (j = 0; j < pairs[i].nblkidxc; j++) {
			xm_dim_t blkidxc = pairs[i].blkidxc[j];
			xm_tensor_read_block(c, blkidxc, bufc2);
			if (aidxc.n > 0 && aidxc.i[0] == 0) {
				xm_tensor_unfold_block(c, blkidxc, aidxc, cidxc,
				    bufc2, bufc1, n);
			} else {
				xm_tensor_unfold_block(c, blkidxc, cidxc, aidxc,
				    bufc2, bufc1, m);
			}
			xm_scalar_axpy(bufc1, 1.0, bufc3, pairs[i].alpha[j],
			    m * n, type);
			if (aidxc.n > 0 && aidxc.i[0] == 0) {
				xm_tensor_fold_block(c, blkidxc, aidxc, cidxc,
				    bufc1, bufc2, n);
			} else {
				xm_tensor_fold_block(c, blkidxc, cidxc, aidxc,
				    bufc1, bufc2, m);
			}
			xm_tensor_write_block(c, blkidxc, bufc2);
		}
	}
done:
	for (i = 0; i < npairs; i++) {
		free(pairs[i].alpha);
		free(pairs[i].blkidxc);
	}
	free(pairs);
	free(canblkc);
	free(bufa1);
}
