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

#define BATCH_BLOCKS_K 128

struct xm_ctx {
	xm_scalar_t alpha, beta;
	xm_tensor_t *a, *b, *c;
	const xm_block_space_t *bsa, *bsb, *bsc;
	xm_dim_t cidxa, aidxa;
	xm_dim_t cidxb, aidxb;
	xm_dim_t cidxc, aidxc;
	size_t nblk_m, nblk_n, nblk_k;
};

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
compute_block(struct xm_ctx *ctx, xm_dim_t blkidxc, xm_scalar_t *buf)
{
	size_t i, m, n, k = 0, nbatched = 0, stride_a, stride_b;
	xm_scalar_t *buf_a, *buf_b, *blkbuf_a, *blkbuf_b;
	xm_scalar_t *blkbuf_c1, *blkbuf_c2, *buf_a_ptr, *buf_b_ptr;
	xm_dim_t blkdimsc = xm_tensor_get_block_dims(ctx->c, blkidxc);
	xm_dim_t blkidxa, blkidxb, nblocksa, nblocksb;
	size_t maxblocksizea, maxblocksizeb, maxblocksizec;
	uintptr_t data_ptr;
	xm_allocator_t *alloca = xm_tensor_get_allocator(ctx->a);
	xm_allocator_t *allocb = xm_tensor_get_allocator(ctx->b);
	xm_allocator_t *allocc = xm_tensor_get_allocator(ctx->c);

	maxblocksizea = xm_block_space_get_largest_block_size(ctx->bsa);
	maxblocksizeb = xm_block_space_get_largest_block_size(ctx->bsb);
	maxblocksizec = xm_block_space_get_largest_block_size(ctx->bsc);
	nblocksa = xm_tensor_get_nblocks(ctx->a);
	nblocksb = xm_tensor_get_nblocks(ctx->b);
	m = xm_dim_dot_mask(&blkdimsc, &ctx->cidxc);
	n = xm_dim_dot_mask(&blkdimsc, &ctx->aidxc);
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
	blkidxa = xm_dim_zero(xm_block_space_get_ndims(ctx->bsa));
	blkidxb = xm_dim_zero(xm_block_space_get_ndims(ctx->bsb));
	xm_dim_set_mask(&blkidxa, &ctx->aidxa, &blkidxc, &ctx->cidxc);
	xm_dim_set_mask(&blkidxb, &ctx->aidxb, &blkidxc, &ctx->aidxc);

	size_t blk_c_size = xm_dim_dot(&blkdimsc);
	xm_allocator_read(allocc, xm_tensor_get_block_data_ptr(
	    ctx->c, blkidxc), blkbuf_c2, blk_c_size * sizeof(xm_scalar_t));
	xm_scalar_t beta = ctx->beta;
	size_t ii, jj;
	if (ctx->aidxc.n > 0 && ctx->aidxc.i[0] == 0) {
		size_t mblk = xm_dim_dot_mask(&blkdimsc, &ctx->cidxc);
		size_t nblk = xm_dim_dot_mask(&blkdimsc, &ctx->aidxc);
		xm_tensor_unfold_block(ctx->c, blkidxc, ctx->aidxc, ctx->cidxc,
		    blkbuf_c2, blkbuf_c1, nblk);
		for (jj = 0; jj < mblk; jj++)
			for (ii = 0; ii < nblk; ii++)
				blkbuf_c1[jj * nblk + ii] *= beta;
	} else {
		size_t mblk = xm_dim_dot_mask(&blkdimsc, &ctx->cidxc);
		size_t nblk = xm_dim_dot_mask(&blkdimsc, &ctx->aidxc);
		xm_tensor_unfold_block(ctx->c, blkidxc, ctx->cidxc, ctx->aidxc,
		    blkbuf_c2, blkbuf_c1, mblk);
		for (jj = 0; jj < nblk; jj++)
			for (ii = 0; ii < mblk; ii++)
				blkbuf_c1[jj * mblk + ii] *= beta;
	}

	if (ctx->alpha == 0.0)
		goto done;
//	for (i = 0; i < ctx->nblk_k; i++) {
//		struct xm_block *blk_a = tensor_get_block(ctx->a, &blkidxa);
//		struct xm_block *blk_b = tensor_get_block(ctx->b, &blkidxb);
//		alphas[i] = ctx->alpha;
//		if (blk_a->type == XM_BLOCK_TYPE_DERIVATIVE &&
//		    blk_b->type == XM_BLOCK_TYPE_DERIVATIVE) {
//
//			alphas[i] = 0.0;
//			alphas[j] *= 2.0;
//		}
//		xm_dim_inc_mask(&blkidxa, &nblocksa, &ctx->cidxa);
//		xm_dim_inc_mask(&blkidxb, &nblocksb, &ctx->cidxb);
//	}
	for (i = 0; i < ctx->nblk_k; i++) {
//		if (alphas[i] == 0.0)
//			goto next;
//		struct xm_block *blk_a = tensor_get_block(ctx->a, blkidxa);
//		struct xm_block *blk_b = tensor_get_block(ctx->b, blkidxb);
		int blka_type = xm_tensor_get_block_type(ctx->a, blkidxa);
		int blkb_type = xm_tensor_get_block_type(ctx->b, blkidxb);

		if (blka_type != XM_BLOCK_TYPE_ZERO &&
		    blkb_type != XM_BLOCK_TYPE_ZERO) {
			size_t mblk, nblk, kblk;
			xm_dim_t blkdimsa = xm_tensor_get_block_dims(
			    ctx->a, blkidxa);
			xm_dim_t blkdimsb = xm_tensor_get_block_dims(
			    ctx->b, blkidxb);
			mblk = xm_dim_dot_mask(&blkdimsa, &ctx->aidxa);
			nblk = xm_dim_dot_mask(&blkdimsb, &ctx->aidxb);
			kblk = xm_dim_dot_mask(&blkdimsa, &ctx->cidxa);

			data_ptr = xm_tensor_get_block_data_ptr(ctx->a,
			    blkidxa);
			size_t blk_a_size = xm_dim_dot(&blkdimsa);
			xm_allocator_read(alloca, data_ptr,
			    blkbuf_a, blk_a_size * sizeof(xm_scalar_t));
			xm_tensor_unfold_block(ctx->a, blkidxa, ctx->cidxa,
			    ctx->aidxa, blkbuf_a, buf_a_ptr,
			    stride_a);
			buf_a_ptr += kblk;

			data_ptr = xm_tensor_get_block_data_ptr(ctx->b,
			    blkidxb);
			size_t blk_b_size = xm_dim_dot(&blkdimsb);
			xm_allocator_read(allocb, data_ptr,
			    blkbuf_b, blk_b_size * sizeof(xm_scalar_t));
			xm_tensor_unfold_block(ctx->b, blkidxb, ctx->cidxb,
			    ctx->aidxb, blkbuf_b, buf_b_ptr,
			    stride_b);
			buf_b_ptr += kblk;

			k += kblk;
			nbatched++;
		}
		if (nbatched >= BATCH_BLOCKS_K ||
		   (i == ctx->nblk_k-1 && nbatched > 0)) {
			if (ctx->aidxc.n > 0 && ctx->aidxc.i[0] == 0) {
				xm_dgemm('T', 'N', (int)n, (int)m, (int)k,
				    ctx->alpha, buf_b, (int)stride_b, buf_a,
				    (int)stride_a, 1.0, blkbuf_c1, (int)n);
			} else {
				xm_dgemm('T', 'N', (int)m, (int)n, (int)k,
				    ctx->alpha, buf_a, (int)stride_a, buf_b,
				    (int)stride_b, 1.0, blkbuf_c1, (int)m);
			}
			k = 0;
			buf_a_ptr = buf_a;
			buf_b_ptr = buf_b;
			nbatched = 0;
		}
//next:
		xm_dim_inc_mask(&blkidxa, &nblocksa, &ctx->cidxa);
		xm_dim_inc_mask(&blkidxb, &nblocksb, &ctx->cidxb);
	}
done:
	if (ctx->aidxc.n > 0 && ctx->aidxc.i[0] == 0) {
		xm_tensor_fold_block(ctx->c, blkidxc, ctx->aidxc, ctx->cidxc,
		    blkbuf_c1, blkbuf_c2, n);
	} else {
		xm_tensor_fold_block(ctx->c, blkidxc, ctx->cidxc, ctx->aidxc,
		    blkbuf_c1, blkbuf_c2, m);
	}
	data_ptr = xm_tensor_get_block_data_ptr(ctx->c, blkidxc);
	xm_allocator_write(allocc, data_ptr,
	    blkbuf_c2, xm_dim_dot(&blkdimsc) * sizeof(xm_scalar_t));
}

void
xm_contract(xm_scalar_t alpha, xm_tensor_t *a, xm_tensor_t *b,
    xm_scalar_t beta, xm_tensor_t *c, const char *idxa, const char *idxb,
    const char *idxc)
{
	struct xm_ctx ctx;
	xm_dim_t cidxa, aidxa, cidxb, aidxb, cidxc, aidxc, *nzblk;
	xm_dim_t nblocksa, nblocksb;
	size_t i, /*si1, si2,*/ size, nnzblk;
	size_t maxblocksizea, maxblocksizeb, maxblocksizec;
//	int sym_k;

	ctx.bsa = xm_tensor_get_block_space(a);
	ctx.bsb = xm_tensor_get_block_space(b);
	ctx.bsc = xm_tensor_get_block_space(c);

	if (strlen(idxa) != xm_block_space_get_ndims(ctx.bsa))
		fatal("bad contraction indices");
	if (strlen(idxb) != xm_block_space_get_ndims(ctx.bsb))
		fatal("bad contraction indices");
	if (strlen(idxc) != xm_block_space_get_ndims(ctx.bsc))
		fatal("bad contraction indices");

	make_masks(idxa, idxb, &cidxa, &cidxb);
	make_masks(idxc, idxa, &cidxc, &aidxa);
	make_masks(idxc, idxb, &aidxc, &aidxb);

	for (i = 0; i < cidxa.n; i++)
		if (!xm_block_space_eq1(ctx.bsa, cidxa.i[i], ctx.bsb,
		    cidxb.i[i]))
			fatal("inconsistent tensor block spaces");
	for (i = 0; i < cidxc.n; i++)
		if (!xm_block_space_eq1(ctx.bsc, cidxc.i[i], ctx.bsa,
		    aidxa.i[i]))
			fatal("inconsistent tensor block spaces");
	for (i = 0; i < aidxc.n; i++)
		if (!xm_block_space_eq1(ctx.bsc, aidxc.i[i], ctx.bsb,
		    aidxb.i[i]))
			fatal("inconsistent tensor block spaces");

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

	if (aidxa.n + cidxa.n != xm_block_space_get_ndims(ctx.bsa))
		fatal("bad indices");
	if (aidxb.n + cidxb.n != xm_block_space_get_ndims(ctx.bsb))
		fatal("bad indices");
	if (aidxc.n + cidxc.n != xm_block_space_get_ndims(ctx.bsc))
		fatal("bad indices");
	if (!(aidxc.n > 0 && aidxc.i[0] == 0) &&
	    !(cidxc.n > 0 && cidxc.i[0] == 0))
		fatal("bad indices");

//	sym_k = 0;
//	si2 = 0;
//	for (si1 = 0; si1 < cidxa.n; si1++) {
//		for (si2 = si1+1; si2 < cidxa.n; si2++) {
//			sym_k = has_k_symmetry(a, cidxa, aidxa, si1, si2) &&
//				has_k_symmetry(b, cidxb, aidxb, si1, si2);
//			if (sym_k)
//				break;
//		}
//		if (sym_k)
//			break;
//	}
//	if (sym_k)
//		set_k_symmetry(a, cidxa, aidxa, si1, si2, 1);

	maxblocksizea = xm_block_space_get_largest_block_size(ctx.bsa);
	maxblocksizeb = xm_block_space_get_largest_block_size(ctx.bsb);
	maxblocksizec = xm_block_space_get_largest_block_size(ctx.bsc);
	nzblk = get_canonical_block_list(ctx.c, &nnzblk);
	size = maxblocksizea * (BATCH_BLOCKS_K + 1) +
	       maxblocksizeb * (BATCH_BLOCKS_K + 1) +
	       maxblocksizec * 2;
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
	for (i = 0; i < nnzblk; i++)
		compute_block(&ctx, nzblk[i], buf);
	free(buf);
}
//	if (sym_k)
//		set_k_symmetry(a, cidxa, aidxa, si1, si2, 0);
	free(nzblk);
}
