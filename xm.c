/*
 * Copyright (c) 2014-2016 Ilya Kaliman <ilya.kaliman@gmail.com>
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
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>
#include <unistd.h>

#ifdef HAVE_BITSTRING_H
#include <bitstring.h>
#else
#include "compat/bitstring.h"
#endif

#include "xm.h"

/** defines a tensor block */
struct xm_block {
	/** pointer to data allocated with an allocator */
	uintptr_t               data_ptr;
	/** block dimensions in scalar elements */
	xm_dim_t                dim;
	/** permutation that is applied to raw data */
	xm_dim_t                permutation;
	/** index of the canonical block if this block is not a source block */
	xm_dim_t                source_idx;
	/** scalar multiplier for the raw data */
	xm_scalar_t             scalar;
	/** specifies whether the block is a source block (boolean) */
	int                     is_source;
	/** specifies whether the block is a zero-block (boolean) */
	int                     is_nonzero;
	/** specifies whether the block is initialized (boolean) */
	int                     is_initialized;
};

/** defines a tensor */
struct xm_tensor {
	/** tensor label string */
	char                   *label;
	/** tensor dimensions in blocks */
	xm_dim_t                dim;
	/** allocator used to allocate tensor data */
	struct xm_allocator    *allocator;
	/** array of all tensor blocks */
	struct xm_block        *blocks;
	/** buffer that is large enough to store any single block data */
	xm_scalar_t            *block_buf;
	/** size of the above buffer */
	size_t                  max_block_size;
};

struct ctx {
	xm_scalar_t alpha;
	struct xm_tensor *a, *b, *c;
	xm_dim_t cidxa, aidxa;
	xm_dim_t cidxb, aidxb;
	xm_dim_t cidxc, aidxc;
	size_t nblk_m, nblk_n, nblk_k;
};

/** helper structure for passing data to worker threads */
struct async {
	struct xm_tensor       *tensor;
	xm_dim_t                blk_idx;
	xm_dim_t                mask_i;
	xm_dim_t                mask_j;
	size_t                  nblk_i;
	size_t                  nblk_j;
	const bitstr_t         *skip_i;
	const bitstr_t         *skip_j;
	xm_scalar_t            *data;
	size_t                  stride;
	xm_scalar_t            *from;
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

#define xm_log_line(msg) \
    xm_log("%s (in %s on line %d)", msg, __func__, __LINE__)
//#define xm_log_perror(msg) \
//    xm_log("%s (%s)", msg, strerror(errno))

/* stream to log to */
static FILE *xm_log_stream = NULL;

void
xm_set_log_stream(FILE *stream)
{
	xm_log_stream = stream;
}

static void
xm_log(const char *fmt, ...)
{
	char buf[1024];
	va_list ap;

	if (xm_log_stream == NULL)
		return;

	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	fprintf(xm_log_stream, "xm_log: %s\n", buf);
}

static void *
xmalloc(size_t size)
{
	void *mem;

	if ((mem = malloc(size)) == NULL) {
		xm_log("out of memory");
		abort();
	}

	return (mem);
}

struct timer {
	char label[128];
	time_t start;
};

static struct timer
timer_start(const char *fmt, ...)
{
	struct timer timer;
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(timer.label, sizeof(timer.label), fmt, ap);
	va_end(ap);

	xm_log("%s...", timer.label);
	timer.start = time(NULL);

	return (timer);
}

static void
timer_stop(struct timer *timer)
{
	time_t total = time(NULL) - timer->start;
	xm_log("%s done in %d sec", timer->label, (int)total);
}

/*
static void
swap_ptr(xm_scalar_t **a, xm_scalar_t **b)
{
	xm_scalar_t *t = *a;
	*a = *b;
	*b = t;
}
*/

xm_dim_t
xm_dim_2(size_t dim1, size_t dim2)
{
	xm_dim_t dim;

	dim.n = 2;
	dim.i[0] = dim1;
	dim.i[1] = dim2;

	return (dim);
}

xm_dim_t
xm_dim_3(size_t dim1, size_t dim2, size_t dim3)
{
	xm_dim_t dim;

	dim.n = 3;
	dim.i[0] = dim1;
	dim.i[1] = dim2;
	dim.i[2] = dim3;

	return (dim);
}

xm_dim_t
xm_dim_4(size_t dim1, size_t dim2, size_t dim3, size_t dim4)
{
	xm_dim_t dim;

	dim.n = 4;
	dim.i[0] = dim1;
	dim.i[1] = dim2;
	dim.i[2] = dim3;
	dim.i[3] = dim4;

	return (dim);
}

xm_dim_t
xm_dim_5(size_t dim1, size_t dim2, size_t dim3, size_t dim4, size_t dim5)
{
	xm_dim_t dim;

	dim.n = 5;
	dim.i[0] = dim1;
	dim.i[1] = dim2;
	dim.i[2] = dim3;
	dim.i[3] = dim4;
	dim.i[4] = dim5;

	return (dim);
}

xm_dim_t
xm_dim_6(size_t dim1, size_t dim2, size_t dim3, size_t dim4, size_t dim5,
    size_t dim6)
{
	xm_dim_t dim;

	dim.n = 6;
	dim.i[0] = dim1;
	dim.i[1] = dim2;
	dim.i[2] = dim3;
	dim.i[3] = dim4;
	dim.i[4] = dim5;
	dim.i[5] = dim6;

	return (dim);
}

xm_dim_t
xm_dim_same(size_t n, size_t dim)
{
	xm_dim_t ret;

	assert(n <= XM_MAX_DIM);

	for (ret.n = 0; ret.n < n; ret.n++)
		ret.i[ret.n] = dim;

	return (ret);
}

xm_dim_t
xm_dim_zero(size_t n)
{
	return (xm_dim_same(n, 0));
}

static int
xm_dim_eq(const xm_dim_t *a, const xm_dim_t *b)
{
	size_t i;

	if (a->n != b->n)
		return (0);
	for (i = 0; i < a->n; i++)
		if (a->i[i] != b->i[i])
			return (0);
	return (1);
}

//static void
//xm_dim_set_mask(xm_dim_t *a, const xm_dim_t *b, const xm_dim_t *mask)
//{
//	size_t i;
//
//	assert(a->n == b->n);
//
//	for (i = 0; i < mask->n; i++)
//		a->i[mask->i[i]] = b->i[mask->i[i]];
//}

static void
xm_dim_set_mask2(xm_dim_t *a, const xm_dim_t *ma,
    const xm_dim_t *b, const xm_dim_t *mb)
{
	size_t i;

	assert(ma->n == mb->n);

	for (i = 0; i < ma->n; i++)
		a->i[ma->i[i]] = b->i[mb->i[i]];
}

static void
xm_dim_zero_mask(xm_dim_t *a, const xm_dim_t *mask)
{
	size_t i;

	for (i = 0; i < mask->n; i++)
		a->i[mask->i[i]] = 0;
}

size_t
xm_dim_dot(const xm_dim_t *dim)
{
	size_t i, ret;

	ret = 1;
	for (i = 0; i < dim->n; i++)
		ret *= dim->i[i];

	return (ret);
}

static size_t
xm_dim_dot_mask(const xm_dim_t *dim, const xm_dim_t *mask)
{
	size_t i, ret;

	ret = 1;
	for (i = 0; i < mask->n; i++)
		ret *= dim->i[mask->i[i]];

	return (ret);
}

xm_dim_t
xm_dim_identity_permutation(size_t n)
{
	xm_dim_t ret;

	assert(n <= XM_MAX_DIM);

	for (ret.n = 0; ret.n < n; ret.n++)
		ret.i[ret.n] = ret.n;

	return (ret);
}

int
xm_dim_less(const xm_dim_t *idx, const xm_dim_t *dim)
{
	size_t i;

	assert(idx->n == dim->n);

	for (i = 0; i < idx->n; i++)
		if (idx->i[i] >= dim->i[i])
			return (0);
	return (1);
}

static size_t
xm_dim_offset(const xm_dim_t *idx, const xm_dim_t *dim)
{
	size_t ret = 0;

	assert(xm_dim_less(idx, dim));

	switch (idx->n) {
	case 6: ret += idx->i[5] * dim->i[4] * dim->i[3] * dim->i[2] * dim->i[1] * dim->i[0];
	case 5: ret += idx->i[4] * dim->i[3] * dim->i[2] * dim->i[1] * dim->i[0];
	case 4: ret += idx->i[3] * dim->i[2] * dim->i[1] * dim->i[0];
	case 3: ret += idx->i[2] * dim->i[1] * dim->i[0];
	case 2: ret += idx->i[1] * dim->i[0];
	case 1: ret += idx->i[0];
	}
	return (ret);
}

//static size_t
//xm_dim_offset_mask(const xm_dim_t *idx, const xm_dim_t *dim,
//    const xm_dim_t *mask)
//{
//	size_t i, offset, power;
//
//	assert(xm_dim_less(idx, dim));
//
//	offset = 0;
//	power = 1;
//	for (i = 0; i < mask->n; i++) {
//		offset += idx->i[mask->i[i]] * power;
//		power *= dim->i[mask->i[i]];
//	}
//	return (offset);
//}

size_t
xm_dim_inc(xm_dim_t *idx, const xm_dim_t *dim)
{
	size_t i, carry = 1;

	assert(dim->n == idx->n);

	for (i = 0; carry && i < idx->n; i++) {
		idx->i[i] += carry;
		carry = idx->i[i] / dim->i[i];
		idx->i[i] %= dim->i[i];
	}

	return (carry);
}

static size_t
xm_dim_inc_mask(xm_dim_t *idx, const xm_dim_t *dim, const xm_dim_t *mask)
{
	size_t i, carry = 1;

	assert(dim->n == idx->n);

	for (i = 0; carry && i < mask->n; i++) {
		idx->i[mask->i[i]] += carry;
		carry = idx->i[mask->i[i]] / dim->i[mask->i[i]];
		idx->i[mask->i[i]] %= dim->i[mask->i[i]];
	}

	return (carry);
}

static xm_dim_t
xm_dim_permute(const xm_dim_t *idx, const xm_dim_t *permutation)
{
	xm_dim_t ret;

	assert(idx->n == permutation->n);

	ret.n = idx->n;
	switch (ret.n) {
	case 6: ret.i[permutation->i[5]] = idx->i[5];
	case 5: ret.i[permutation->i[4]] = idx->i[4];
	case 4: ret.i[permutation->i[3]] = idx->i[3];
	case 3: ret.i[permutation->i[2]] = idx->i[2];
	case 2: ret.i[permutation->i[1]] = idx->i[1];
	case 1: ret.i[permutation->i[0]] = idx->i[0];
	}
	return (ret);
}

static xm_dim_t
xm_dim_permute_rev(const xm_dim_t *idx, const xm_dim_t *permutation)
{
	xm_dim_t ret;

	assert(idx->n == permutation->n);

	ret.n = idx->n;
	switch (ret.n) {
	case 6: ret.i[5] = idx->i[permutation->i[5]];
	case 5: ret.i[4] = idx->i[permutation->i[4]];
	case 4: ret.i[3] = idx->i[permutation->i[3]];
	case 3: ret.i[2] = idx->i[permutation->i[2]];
	case 2: ret.i[1] = idx->i[permutation->i[1]];
	case 1: ret.i[0] = idx->i[permutation->i[0]];
	}
	return (ret);
}

struct xm_tensor *
xm_tensor_create(struct xm_allocator *allocator, const xm_dim_t *dim,
    const char *label)
{
	struct xm_tensor *tensor;
	size_t i, size;

	assert(dim->n >= 1 && dim->n <= XM_MAX_DIM);

	if ((tensor = calloc(1, sizeof(*tensor))) == NULL) {
		xm_log_line("out of memory");
		return (NULL);
	}

	size = xm_dim_dot(dim);

	if ((tensor->blocks = calloc(size, sizeof(*tensor->blocks))) == NULL) {
		xm_log_line("out of memory");
		free(tensor);
		return (NULL);
	}

	for (i = 0; i < size; i++)
		tensor->blocks[i].data_ptr = XM_NULL_PTR;

	tensor->label = strdup(label ? label : "");
	tensor->dim = *dim;
	tensor->allocator = allocator;

	return (tensor);
}

const char *
xm_tensor_get_label(const struct xm_tensor *tensor)
{
	assert(tensor);

	return (tensor->label);
}

static struct xm_block *
xm_tensor_get_block(const struct xm_tensor *tensor, const xm_dim_t *idx)
{
	size_t offset;

	assert(tensor);
	assert(idx);

	offset = xm_dim_offset(idx, &tensor->dim);
	return (tensor->blocks + offset);
}

void
xm_tensor_copy_data(struct xm_tensor *dst, const struct xm_tensor *src)
{
	struct xm_block *dstblk, *srcblk;
	size_t i, nblk, size;

	assert(xm_tensor_is_initialized(dst));
	assert(xm_tensor_is_initialized(src));
	assert(xm_dim_eq(&dst->dim, &src->dim));

	nblk = xm_dim_dot(&dst->dim);

	for (i = 0; i < nblk; i++) {
		dstblk = &dst->blocks[i];
		srcblk = &src->blocks[i];
		assert(srcblk->is_source == dstblk->is_source);
		assert(srcblk->is_nonzero == dstblk->is_nonzero);
		assert(xm_dim_eq(&srcblk->dim, &dstblk->dim));
		if (dstblk->is_source) {
			size = xm_dim_dot(&dstblk->dim) * sizeof(xm_scalar_t);
			assert(srcblk->data_ptr != XM_NULL_PTR);
			xm_allocator_read(src->allocator, srcblk->data_ptr,
			    dst->block_buf, size);
			assert(dstblk->data_ptr != XM_NULL_PTR);
			xm_allocator_write(dst->allocator, dstblk->data_ptr,
			    dst->block_buf, size);
		}
	}
}

xm_scalar_t
xm_tensor_get_element(struct xm_tensor *tensor, const xm_dim_t *blk_i,
    const xm_dim_t *el_i)
{
	struct xm_block *block;
	xm_dim_t idx, dim_p;
	size_t el_off, size_bytes;

	block = xm_tensor_get_block(tensor, blk_i);
	if (!block->is_nonzero)
		return (0.0);
	size_bytes = xm_dim_dot(&block->dim) * sizeof(xm_scalar_t);
	assert(block->data_ptr != XM_NULL_PTR);
	xm_allocator_read(tensor->allocator, block->data_ptr,
	    tensor->block_buf, size_bytes);
	idx = xm_dim_permute(el_i, &block->permutation);
	dim_p = xm_dim_permute(&block->dim, &block->permutation);
	el_off = xm_dim_offset(&idx, &dim_p);
	return (block->scalar * tensor->block_buf[el_off]);
}

static void
xm_tensor_get_idx(struct xm_tensor *tensor, const xm_dim_t *abs_idx,
    xm_dim_t *blk_idx, xm_dim_t *el_idx)
{
	struct xm_block *block;
	xm_dim_t abs_dim, idx;
	size_t dim_i, blk_i, next;

	*blk_idx = xm_dim_zero(tensor->dim.n);
	*el_idx = xm_dim_zero(tensor->dim.n);

	abs_dim = xm_dim_zero(tensor->dim.n);
	for (dim_i = 0; dim_i < tensor->dim.n; dim_i++) {
		idx = xm_dim_zero(tensor->dim.n);
		for (blk_i = 0; blk_i < tensor->dim.i[dim_i]; blk_i++) {
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
		if (blk_i == tensor->dim.i[dim_i])
			assert(0);
	}
}

xm_scalar_t
xm_tensor_get_abs_element(struct xm_tensor *tensor, const xm_dim_t *idx)
{
	xm_dim_t blk_i, el_i;

	xm_tensor_get_idx(tensor, idx, &blk_i, &el_i);
	return (xm_tensor_get_element(tensor, &blk_i, &el_i));
}

int
xm_tensor_block_is_nonzero(const struct xm_tensor *tensor, const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->is_nonzero);
}

int
xm_tensor_block_is_initialized(const struct xm_tensor *tensor,
    const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->is_initialized);
}

xm_dim_t
xm_tensor_get_block_dim(const struct xm_tensor *tensor, const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->dim);
}

uintptr_t
xm_tensor_get_block_data_ptr(const struct xm_tensor *tensor,
    const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->data_ptr);
}

xm_dim_t
xm_tensor_get_block_permutation(const struct xm_tensor *tensor,
    const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->permutation);
}

xm_scalar_t
xm_tensor_get_block_scalar(const struct xm_tensor *tensor,
    const xm_dim_t *idx)
{
	struct xm_block *block;

	block = xm_tensor_get_block(tensor, idx);
	return (block->scalar);
}

static void
xm_tensor_set_block_buf(struct xm_tensor *a, const xm_dim_t *blkdim)
{
	size_t size = xm_dim_dot(blkdim);

	if (size > a->max_block_size) {
		if ((a->block_buf = realloc(a->block_buf,
		    size * sizeof(xm_scalar_t))) == NULL) {
			xm_log_line("out of memory");
			abort();
		}
		a->max_block_size = size;
	}
}

void
xm_tensor_reset_block(struct xm_tensor *tensor, const xm_dim_t *idx)
{
	struct xm_block *block;

	assert(tensor != NULL);
	assert(idx != NULL);

	block = xm_tensor_get_block(tensor, idx);
	memset(block, 0, sizeof(*block));
	block->data_ptr = XM_NULL_PTR;
}

void
xm_tensor_set_zero_block(struct xm_tensor *tensor, const xm_dim_t *idx,
    const xm_dim_t *blkdim)
{
	struct xm_block *block;

	assert(tensor != NULL);
	assert(idx != NULL);
	assert(blkdim != NULL);

	block = xm_tensor_get_block(tensor, idx);
	assert(!block->is_initialized);

	block->source_idx = *idx;
	block->dim = *blkdim;
	block->data_ptr = XM_NULL_PTR;
	block->permutation = xm_dim_identity_permutation(blkdim->n);
	block->scalar = 1.0;
	block->is_source = 0;
	block->is_nonzero = 0;
	block->is_initialized = 1;

	xm_tensor_set_block_buf(tensor, blkdim);
}

void
xm_tensor_set_source_block(struct xm_tensor *tensor, const xm_dim_t *idx,
    const xm_dim_t *blkdim, uintptr_t data_ptr)
{
	struct xm_block *block;

	assert(tensor != NULL);
	assert(idx != NULL);
	assert(blkdim != NULL);
	assert(data_ptr != XM_NULL_PTR);

	block = xm_tensor_get_block(tensor, idx);
	assert(!block->is_initialized);

	block->source_idx = *idx;
	block->dim = *blkdim;
	block->data_ptr = data_ptr;
	block->permutation = xm_dim_identity_permutation(blkdim->n);
	block->scalar = 1.0;
	block->is_source = 1;
	block->is_nonzero = 1;
	block->is_initialized = 1;

	xm_tensor_set_block_buf(tensor, blkdim);
}

void
xm_tensor_set_block(struct xm_tensor *tensor, const xm_dim_t *idx,
    const xm_dim_t *source_idx, const xm_dim_t *permutation,
    xm_scalar_t scalar)
{
	struct xm_block *block, *source_block;
	xm_dim_t blkdim;

	assert(tensor != NULL);
	assert(idx != NULL);
	assert(source_idx != NULL);
	assert(permutation != NULL);

	source_block = xm_tensor_get_block(tensor, source_idx);
	assert(source_block->is_initialized);
	assert(source_block->is_source);
	assert(source_block->is_nonzero);

	block = xm_tensor_get_block(tensor, idx);
	assert(!block->is_initialized);

	blkdim = xm_dim_permute_rev(&source_block->dim, permutation);
	block->source_idx = *source_idx;
	block->dim = blkdim;
	block->data_ptr = source_block->data_ptr;
	block->permutation = *permutation;
	block->scalar = scalar;
	block->is_source = 0;
	block->is_nonzero = 1;
	block->is_initialized = 1;

	xm_tensor_set_block_buf(tensor, &blkdim);
}

int
xm_tensor_is_initialized(const struct xm_tensor *tensor)
{
	size_t i, dim;

	assert(tensor);

	dim = xm_dim_dot(&tensor->dim);

	for (i = 0; i < dim; i++)
		if (!tensor->blocks[i].is_initialized)
			return (0);
	return (1);
}

xm_dim_t
xm_tensor_get_dim(const struct xm_tensor *tensor)
{
	assert(tensor);

	return (tensor->dim);
}

xm_dim_t
xm_tensor_get_abs_dim(const struct xm_tensor *tensor)
{
	struct xm_block *block;
	xm_dim_t abs_dim, idx;
	size_t dim_i, blk_i;

	assert(xm_tensor_is_initialized(tensor));

	abs_dim = xm_dim_zero(tensor->dim.n);
	for (dim_i = 0; dim_i < tensor->dim.n; dim_i++) {
		idx = xm_dim_zero(tensor->dim.n);
		for (blk_i = 0; blk_i < tensor->dim.i[dim_i]; blk_i++) {
			idx.i[dim_i] = blk_i;
			block = xm_tensor_get_block(tensor, &idx);
			abs_dim.i[dim_i] += block->dim.i[dim_i];
		}
	}

	return (abs_dim);
}

void
xm_tensor_free(struct xm_tensor *tensor)
{
	if (tensor) {
		free(tensor->blocks);
		free(tensor->label);
		free(tensor->block_buf);
		free(tensor);
	}
}

//static int
//skip_idx(xm_dim_t *idx, struct xm_tensor *a, const xm_dim_t *mask,
//    const bitstr_t *skip)
//{
//	size_t i = xm_dim_offset_mask(idx, &a->dim, mask);
//
//	for (; bit_test(skip, i); i++)
//		if (xm_dim_inc_mask(idx, &a->dim, mask))
//			return (1);
//
//	return (0);
//}

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

#if 0
static void
tensor_get_submatrix(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    xm_scalar_t *data, const bitstr_t *skip_i, const bitstr_t *skip_j,
    size_t stride, xm_scalar_t *from)
{
	struct xm_block *block;
	xm_dim_t blk_idx2;
	size_t i, j, block_size_i, block_size_j = 0;
	xm_scalar_t *ptr, *buf = from;

	if (from == NULL) {
		/* accessed from multiple threads */
		if ((buf = malloc(tensor->max_block_size * sizeof(xm_scalar_t))) == NULL) {
			xm_log_line("out of memory");
			abort();
		}
	}

	blk_idx2 = blk_idx;

	for (j = 0; j < nblk_j; j++) {
		skip_idx(&blk_idx, tensor, &mask_j, skip_j);

		xm_dim_set_mask(&blk_idx, &blk_idx2, &mask_i);
		ptr = data;
		for (i = 0; i < nblk_i; i++) {
			skip_idx(&blk_idx, tensor, &mask_i, skip_i);

			block = xm_tensor_get_block(tensor, &blk_idx);
			block_size_i = xm_dim_dot_mask(&block->dim, &mask_i);
			block_size_j = xm_dim_dot_mask(&block->dim, &mask_j);

			if (block->is_nonzero) {
				if (from == NULL) {
					assert(block->data_ptr != XM_NULL_PTR);
					xm_allocator_read(tensor->allocator,
					    block->data_ptr, buf,
					    block_size_i * block_size_j *
					    sizeof(xm_scalar_t));
				}
			} else {
				memset(buf, 0, block_size_i *
				    block_size_j * sizeof(xm_scalar_t));
			}
			block_get_matrix(block, mask_i, mask_j,
			    block_size_i, block_size_j,
			    buf, ptr, stride);
			if (from)
				buf += block_size_i * block_size_j;
			ptr += block_size_i;

			if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_i))
				assert(i == nblk_i - 1);
		}
		data += block_size_j * stride;

		if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_j))
			assert(j == nblk_j - 1);
	}
	if (from == NULL)
		free(buf);
}

static void *
async_tensor_get_submatrix_routine(void *arg)
{
	struct async *async = (struct async *)arg;
	struct timer timer;

	timer = timer_start("%s(%s)", __func__, async->tensor->label);
	tensor_get_submatrix(async->tensor, async->blk_idx, async->mask_i,
	    async->mask_j, async->nblk_i, async->nblk_j, async->data,
	    async->skip_i, async->skip_j, async->stride,
	    async->from);
	timer_stop(&timer);

	free(async);
	return (NULL);
}

static pthread_t
async_tensor_get_submatrix(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    const bitstr_t *skip_i, const bitstr_t *skip_j, size_t stride,
    xm_scalar_t *from, xm_scalar_t *to)
{
	struct async *async;
	pthread_t thread;

	if ((async = calloc(1, sizeof(*async))) == NULL) {
		xm_log_line("out of memory");
		abort();
	}

	async->tensor = tensor;
	async->blk_idx = blk_idx;
	async->mask_i = mask_i;
	async->mask_j = mask_j;
	async->nblk_i = nblk_i;
	async->nblk_j = nblk_j;
	async->data = to;
	async->skip_i = skip_i;
	async->skip_j = skip_j;
	async->stride = stride;
	async->from = from;

	if (pthread_create(&thread, NULL,
	    async_tensor_get_submatrix_routine, async)) {
		xm_log_perror("pthread_create");
		abort();
	}

	return (thread);
}
#endif

static void
block_set_matrix(struct xm_block *block, xm_dim_t mask_i, xm_dim_t mask_j,
    size_t block_size_i, size_t block_size_j, xm_scalar_t *data,
    size_t stride, xm_scalar_t *buf, struct xm_allocator *allocator)
{
	xm_dim_t el_dim, el_i;
	size_t ii, jj, offset;

	assert(block->is_source);
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

#if 0
static void
tensor_set_submatrix(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    const bitstr_t *skip_i, const bitstr_t *skip_j, xm_scalar_t *data,
    size_t stride)
{
	struct xm_block *block;
	xm_dim_t blk_idx2;
	size_t i, j, block_size_i, block_size_j = 0;
	xm_scalar_t *ptr, *buf;

	/* accessed from multiple threads */
	if ((buf = malloc(tensor->max_block_size * sizeof(xm_scalar_t))) == NULL) {
		xm_log_line("out of memory");
		abort();
	}

	blk_idx2 = blk_idx;

	for (j = 0; j < nblk_j; j++) {
		skip_idx(&blk_idx, tensor, &mask_j, skip_j);

		xm_dim_set_mask(&blk_idx, &blk_idx2, &mask_i);
		ptr = data;
		for (i = 0; i < nblk_i; i++) {
			skip_idx(&blk_idx, tensor, &mask_i, skip_i);

			block = xm_tensor_get_block(tensor, &blk_idx);
			block_size_i = xm_dim_dot_mask(&block->dim, &mask_i);
			block_size_j = xm_dim_dot_mask(&block->dim, &mask_j);

			block_set_matrix(block, mask_i, mask_j,
			    block_size_i, block_size_j,
			    ptr, stride, buf, tensor->allocator);
			ptr += block_size_i;

			if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_i))
				assert(i == nblk_i - 1);
		}
		data += block_size_j * stride;

		if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_j))
			assert(j == nblk_j - 1);
	}
	free(buf);
}

static void *
async_tensor_set_submatrix_routine(void *arg)
{
	struct async *async = (struct async *)arg;
	struct timer timer;

	timer = timer_start("%s(%s)", __func__, async->tensor->label);
	tensor_set_submatrix(async->tensor, async->blk_idx, async->mask_i,
	    async->mask_j, async->nblk_i, async->nblk_j,
	    async->skip_i, async->skip_j, async->data, async->stride);
	timer_stop(&timer);

	free(async);
	return (NULL);
}

static pthread_t
async_tensor_set_submatrix(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    const bitstr_t *skip_i, const bitstr_t *skip_j,
    size_t stride, xm_scalar_t *data)
{
	struct async *async;
	pthread_t thread;

	if ((async = calloc(1, sizeof(*async))) == NULL) {
		xm_log_line("out of memory");
		abort();
	}

	async->tensor = tensor;
	async->blk_idx = blk_idx;
	async->mask_i = mask_i;
	async->mask_j = mask_j;
	async->nblk_i = nblk_i;
	async->nblk_j = nblk_j;
	async->data = data;
	async->skip_i = skip_i;
	async->skip_j = skip_j;
	async->stride = stride;

	if (pthread_create(&thread, NULL,
	    async_tensor_set_submatrix_routine, async)) {
		xm_log_perror("pthread_create");
		abort();
	}

	return (thread);
}

static int
add_chunk_blocks(struct xm_tensor *tensor, xm_dim_t *blk_idx, xm_dim_t mask,
    const bitstr_t *skip, size_t max_cs)
{
	struct xm_block *block;
	size_t cs;

	cs = 0;
	for (;;) {
		if (skip_idx(blk_idx, tensor, &mask, skip))
			return (1);
		block = xm_tensor_get_block(tensor, blk_idx);
		cs += xm_dim_dot_mask(&block->dim, &mask);
		if (cs > max_cs)
			return (0);
		if (xm_dim_inc_mask(blk_idx, &tensor->dim, &mask))
			return (1);
	}
}

static size_t
get_chunk_blocks(struct xm_tensor *tensor, xm_dim_t blk_idx, xm_dim_t mask,
    const bitstr_t *skip, size_t max_cs)
{
	struct xm_block *block;
	size_t i, cs;

	for (i = 0, cs = 0; ; i++) {
		if (skip_idx(&blk_idx, tensor, &mask, skip))
			break;
		block = xm_tensor_get_block(tensor, &blk_idx);
		cs += xm_dim_dot_mask(&block->dim, &mask);
		if (cs > max_cs)
			break;
		if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask))
			return (i + 1);
	}
	return (i);
}

static size_t
get_min_chunk_size(struct xm_tensor *a, xm_dim_t mask, const bitstr_t *skip)
{
	struct xm_block *block;
	xm_dim_t blk_idx;
	size_t size, max_size;

	blk_idx = xm_dim_zero(a->dim.n);
	max_size = 0;

	for (;;) {
		if (skip_idx(&blk_idx, a, &mask, skip))
			break;
		block = xm_tensor_get_block(a, &blk_idx);
		size = xm_dim_dot_mask(&block->dim, &mask);
		if (size > max_size)
			max_size = size;
		if (xm_dim_inc_mask(&blk_idx, &a->dim, &mask))
			break;
	}
	return (max_size);
}

static size_t
get_chunk_size(struct xm_tensor *tensor, xm_dim_t blk_idx, xm_dim_t mask,
    const bitstr_t *skip, size_t max_cs)
{
	struct xm_block *block;
	size_t cs, size;

	cs = 0;
	for (;;) {
		if (skip_idx(&blk_idx, tensor, &mask, skip))
			break;
		block = xm_tensor_get_block(tensor, &blk_idx);
		size = xm_dim_dot_mask(&block->dim, &mask);
		if (cs + size > max_cs)
			break;
		cs += size;
		if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask))
			break;
	}
	return (cs);
}

static void
make_skip(bitstr_t *skip, struct xm_tensor *c, xm_dim_t cidxc, xm_dim_t aidxc)
{
	struct xm_block *block;
	xm_dim_t blk_idx;
	size_t i, j, nblk_i, nblk_j;

	nblk_i = xm_dim_dot_mask(&c->dim, &cidxc);
	nblk_j = xm_dim_dot_mask(&c->dim, &aidxc);

	blk_idx = xm_dim_zero(c->dim.n);
	for (i = 0; i < nblk_i; i++) {
		xm_dim_zero_mask(&blk_idx, &aidxc);
		for (j = 0; j < nblk_j; j++) {
			block = xm_tensor_get_block(c, &blk_idx);
			if (block->is_source) {
				bit_clear(skip, i);
				break;
			}
			xm_dim_inc_mask(&blk_idx, &c->dim, &aidxc);
		}
		xm_dim_inc_mask(&blk_idx, &c->dim, &cidxc);
	}
}

static void
set_skip_zero(bitstr_t *skip, struct xm_tensor *a, xm_dim_t cidxa,
    xm_dim_t aidxa)
{
	struct xm_block *block;
	xm_dim_t blk_idx;
	size_t i, j, nblk_i, nblk_j;
	int all_zero;

	nblk_i = xm_dim_dot_mask(&a->dim, &cidxa);
	nblk_j = xm_dim_dot_mask(&a->dim, &aidxa);

	blk_idx = xm_dim_zero(a->dim.n);
	for (i = 0; i < nblk_i; i++) {
		xm_dim_zero_mask(&blk_idx, &aidxa);
		all_zero = 1;
		for (j = 0; j < nblk_j; j++) {
			block = xm_tensor_get_block(a, &blk_idx);
			if (block->is_nonzero) {
				all_zero = 0;
				break;
			}
			xm_dim_inc_mask(&blk_idx, &a->dim, &aidxa);
		}
		if (all_zero)
			bit_set(skip, i);
		xm_dim_inc_mask(&blk_idx, &a->dim, &cidxa);
	}
}

static size_t
count_zero_bits(const bitstr_t *bits, size_t nbits)
{
	size_t i, cnt;

	for (i = 0, cnt = 0; i < nbits; i++)
		if (!bit_test(bits, i))
			cnt++;

	return (cnt);
}

static void
calc_max_chunk_size(size_t m, size_t n, size_t k, size_t *cs_m, size_t *cs_n)
{
	size_t size, cs_m2, cs_n2;

	size = get_buffer_size() / sizeof(xm_scalar_t);

	/* both a and b fit in buffer */
	cs_m2 = m;
	cs_n2 = n;
	if (k * (cs_m2 + cs_n2) + cs_m2 * cs_n2 <= size)
		goto done;

	/* a or b fits in buffer */
	if (m < n) {
		cs_m2 = m;
		if (size > cs_m2 * k) {
			cs_n2 = (size - cs_m2 * k) / (4 * k + 3 * cs_m2);
			if (cs_n2 >= cs_m2 / 4 && cs_n2 >= *cs_n)
				goto done;
		}
	} else {
		cs_n2 = n;
		if (size > cs_n2 * k) {
			cs_m2 = (size - cs_n2 * k) / (4 * k + 3 * cs_n2);
			if (cs_m2 >= cs_n2 / 4 && cs_m2 >= *cs_m)
				goto done;
		}
	}

	/* neither a nor b fits in buffer */
	cs_m2 = *cs_m;
	cs_n2 = *cs_n;
	while (4 * k * (cs_m2+1+cs_n2+1) + 3 * (cs_m2+1) * (cs_n2+1) < size) {
		cs_m2++;
		cs_n2++;
	}
done:
	assert(cs_m2 <= m);
	assert(cs_n2 <= n);
	assert(cs_m2 >= *cs_m);
	assert(cs_n2 >= *cs_n);
	*cs_m = cs_m2;
	*cs_n = cs_n2;
}

static void
tensor_prefetch(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    const bitstr_t *skip_i, const bitstr_t *skip_j, xm_scalar_t *data)
{
	struct xm_block *block;
	xm_dim_t blk_idx2;
	size_t i, j, size_bytes, size;

	blk_idx2 = blk_idx;

	for (j = 0; j < nblk_j; j++) {
		skip_idx(&blk_idx, tensor, &mask_j, skip_j);

		xm_dim_set_mask(&blk_idx, &blk_idx2, &mask_i);
		for (i = 0; i < nblk_i; i++) {
			skip_idx(&blk_idx, tensor, &mask_i, skip_i);

			block = xm_tensor_get_block(tensor, &blk_idx);
			size = xm_dim_dot(&block->dim);
			if (block->is_nonzero) {
				size_bytes = size * sizeof(xm_scalar_t);
				assert(block->data_ptr != XM_NULL_PTR);
				xm_allocator_read(tensor->allocator,
				    block->data_ptr, data, size_bytes);
			}
			data += size;

			if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_i))
				assert(i == nblk_i - 1);
		}

		if (xm_dim_inc_mask(&blk_idx, &tensor->dim, &mask_j))
			assert(j == nblk_j - 1);
	}
}

static void *
async_tensor_prefetch_routine(void *arg)
{
	struct async *async = (struct async *)arg;
	struct timer timer;

	timer = timer_start("%s(%s)", __func__, async->tensor->label);
	tensor_prefetch(async->tensor, async->blk_idx,
	    async->mask_i, async->mask_j, async->nblk_i, async->nblk_j,
	    async->skip_i, async->skip_j, async->data);
	timer_stop(&timer);

	free(async);
	return (NULL);
}

static pthread_t
async_tensor_prefetch(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    const bitstr_t *skip_i, const bitstr_t *skip_j, xm_scalar_t *data)
{
	struct async *async;
	pthread_t thread;

	if ((async = calloc(1, sizeof(*async))) == NULL) {
		xm_log_line("out of memory");
		abort();
	}

	async->tensor = tensor;
	async->blk_idx = blk_idx;
	async->mask_i = mask_i;
	async->mask_j = mask_j;
	async->nblk_i = nblk_i;
	async->nblk_j = nblk_j;
	async->skip_i = skip_i;
	async->skip_j = skip_j;
	async->data = data;

	if (pthread_create(&thread, NULL,
	    async_tensor_prefetch_routine, async)) {
		xm_log_perror("pthread_create");
		abort();
	}

	return (thread);
}

static pthread_t
async_tensor_prefetch_next(struct xm_tensor *tensor, xm_dim_t blk_idx,
    xm_dim_t mask_i, xm_dim_t mask_j, size_t nblk_i, size_t nblk_j,
    const bitstr_t *skip_i, const bitstr_t *skip_j, xm_scalar_t *data,
    size_t max_cs_m, int wrap)
{
	int done;

	done = add_chunk_blocks(tensor, &blk_idx, mask_j,
	    skip_j, max_cs_m);
	if (wrap || !done) {
		nblk_j = get_chunk_blocks(tensor, blk_idx, mask_j,
		    skip_j, max_cs_m);
		return (async_tensor_prefetch(tensor, blk_idx, mask_i, mask_j,
		    nblk_i, nblk_j, skip_i, skip_j, data));
	}
	return (pthread_self());
}
#endif

static void
check_block_consistency(struct xm_tensor *t1, struct xm_tensor *t2,
    xm_dim_t mask1, xm_dim_t mask2)
{
	struct xm_block *blk1, *blk2;
	xm_dim_t idx1, idx2;
	size_t i, n, dim1, dim2;

	n = xm_dim_dot_mask(&t1->dim, &mask1);
	assert(n == xm_dim_dot_mask(&t2->dim, &mask2));

	idx1 = xm_dim_zero(t1->dim.n);
	idx2 = xm_dim_zero(t2->dim.n);
	for (i = 0; i < n; i++) {
		blk1 = xm_tensor_get_block(t1, &idx1);
		blk2 = xm_tensor_get_block(t2, &idx2);
		dim1 = xm_dim_dot_mask(&blk1->dim, &mask1);
		dim2 = xm_dim_dot_mask(&blk2->dim, &mask2);
		assert(dim1 == dim2);
		xm_dim_inc_mask(&idx1, &t1->dim, &mask1);
		xm_dim_inc_mask(&idx2, &t2->dim, &mask2);
	}
}

static void
parse_idx(const char *str1, const char *str2, xm_dim_t *mask1, xm_dim_t *mask2)
{
	size_t len1, len2, i, j;

	len1 = strlen(str1);
	len2 = strlen(str2);

	mask1->n = 0;
	mask2->n = 0;
	for (i = 0; i < len1; i++) {
		for (j = 0; j < len2; j++) {
			if (str1[i] == str2[j]) {
				mask1->i[mask1->n++] = i;
				mask2->i[mask2->n++] = j;
			}
		}
	}
}

static int
has_k_symmetry(struct xm_tensor *a, xm_dim_t cidxa, xm_dim_t aidxa,
    size_t si1, size_t si2)
{
	struct xm_block *blk, *blk2;
	xm_dim_t idx, idx2;
	size_t i, j, nblk_i, nblk_j;

	if (a->dim.i[cidxa.i[si1]] != a->dim.i[cidxa.i[si2]])
		return (0);

	nblk_i = xm_dim_dot_mask(&a->dim, &cidxa);
	nblk_j = xm_dim_dot_mask(&a->dim, &aidxa);

	idx = xm_dim_zero(a->dim.n);
	for (i = 0; i < nblk_i; i++) {
		xm_dim_zero_mask(&idx, &aidxa);
		for (j = 0; j < nblk_j; j++) {
			blk = xm_tensor_get_block(a, &idx);
			if (idx.i[cidxa.i[si1]] != idx.i[cidxa.i[si2]]) {
				idx2 = idx;
				idx2.i[cidxa.i[si1]] = idx.i[cidxa.i[si2]];
				idx2.i[cidxa.i[si2]] = idx.i[cidxa.i[si1]];
				blk2 = xm_tensor_get_block(a, &idx2);
				if (blk->is_nonzero || blk2->is_nonzero) {
					if (!xm_dim_eq(&blk->source_idx,
						       &blk2->source_idx))
						return (0);
				}
			}
			xm_dim_inc_mask(&idx, &a->dim, &aidxa);
		}
		xm_dim_inc_mask(&idx, &a->dim, &cidxa);
	}

	return (1);
}

static void
set_k_symmetry(struct xm_tensor *a, xm_dim_t cidxa, xm_dim_t aidxa,
    size_t si1, size_t si2, int enable)
{
	struct xm_block *blk;
	xm_dim_t idx;
	size_t nblk_i, nblk_j, i, j;

	nblk_i = xm_dim_dot_mask(&a->dim, &aidxa);
	nblk_j = xm_dim_dot_mask(&a->dim, &cidxa);

	idx = xm_dim_zero(a->dim.n);
	for (i = 0; i < nblk_i; i++) {
		xm_dim_zero_mask(&idx, &cidxa);
		for (j = 0; j < nblk_j; j++) {
			blk = xm_tensor_get_block(a, &idx);
			if (idx.i[cidxa.i[si1]] < idx.i[cidxa.i[si2]])
				blk->scalar *= enable ? 2.0 : 0.5;
			else if (idx.i[cidxa.i[si1]] > idx.i[cidxa.i[si2]]) {
				blk->is_nonzero = !enable &&
				    blk->data_ptr != XM_NULL_PTR;
			}
			xm_dim_inc_mask(&idx, &a->dim, &cidxa);
		}
		xm_dim_inc_mask(&idx, &a->dim, &aidxa);
	}
}

#if 0
static int
xm_do_contract(xm_scalar_t alpha, struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, xm_dim_t cidxa, xm_dim_t aidxa,
    xm_dim_t cidxb, xm_dim_t aidxb, xm_dim_t cidxc, xm_dim_t aidxc,
    const bitstr_t *skip_m, const bitstr_t *skip_n, const bitstr_t *skip_k,
    xm_scalar_t *buf)
{
	xm_scalar_t *blk_a1, *blk_a2, *blk_a3, *blk_a4;
	xm_scalar_t *blk_b1, *blk_b2, *blk_b3, *blk_b4;
	xm_scalar_t *blk_c1, *blk_c2, *blk_c3;
	int done_a, done_b, split_a, split_b;
	pthread_t thr_a1, thr_a2, thr_b1, thr_b2, thr_c1, thr_c2;
	size_t m, n, k, max_cs_m, max_cs_n;
	size_t nblk_k, blk_cs_m, blk_cs_n;
	size_t size, size_a, size_b, size_c;
	struct timer gemm_timer;
	xm_dim_t blk_ia, blk_ib, blk_ic, blk_ic2;

	blk_ia = xm_dim_zero(a->dim.n);
	blk_ib = xm_dim_zero(b->dim.n);
	blk_ic = xm_dim_zero(c->dim.n);

	m = get_chunk_size(c, blk_ic, cidxc, skip_m, ULONG_MAX);
	n = get_chunk_size(c, blk_ic, aidxc, skip_n, ULONG_MAX);
	k = get_chunk_size(a, blk_ia, cidxa, skip_k, ULONG_MAX);
	if (m == 0 || n == 0 || k == 0)
		return (XM_RESULT_SUCCESS);

	max_cs_m = get_min_chunk_size(c, cidxc, skip_m);
	max_cs_n = get_min_chunk_size(c, aidxc, skip_n);
	calc_max_chunk_size(m, n, k, &max_cs_m, &max_cs_n);
	nblk_k = count_zero_bits(skip_k, xm_dim_dot_mask(&a->dim, &cidxa));

	split_a = max_cs_m < m;
	split_b = max_cs_n < n;
	size_a = k * max_cs_m;
	size_b = k * max_cs_n;
	size_c = max_cs_m * max_cs_n;
	size = size_a + size_b + size_c;
	if (split_a)
		size += 3 * size_a;
	if (split_b)
		size += 3 * size_b;
	if (split_a || split_b)
		size += 2 * size_c;

	xm_log("m=%zu n=%zu k=%zu max_cs_m=%zu max_cs_n=%zu",
	    m, n, k, max_cs_m, max_cs_n);
	if (size * sizeof(xm_scalar_t) > get_buffer_size()) {
		xm_log_line("memory buffer is too small");
		return (XM_RESULT_BUFFER_TOO_SMALL);
	}

	blk_a1 = buf;                                /* for dgemm A */
	blk_a2 = split_a ? blk_a1 + size_a : blk_a1; /* permute to */
	blk_a3 = split_a ? blk_a2 + size_a : blk_a2; /* permute from */
	blk_a4 = split_a ? blk_a3 + size_a : blk_a2; /* prefetch */
	blk_b1 = blk_a4 + size_a;                    /* for dgemm B */
	blk_b2 = split_b ? blk_b1 + size_b : blk_b1; /* permute to */
	blk_b3 = split_b ? blk_b2 + size_b : blk_b2; /* permute from */
	blk_b4 = split_b ? blk_b3 + size_b : blk_b2; /* prefetch */
	blk_c1 = blk_b4 + size_b;                    /* for dgemm C */
	blk_c2 = split_a || split_b ? blk_c1 + size_c : blk_c1; /* prefetch */
	blk_c3 = split_a || split_b ? blk_c2 + size_c : blk_c1; /* write back */

	blk_cs_m = get_chunk_blocks(c, blk_ic, cidxc, skip_m, max_cs_m);
	blk_cs_n = get_chunk_blocks(c, blk_ic, aidxc, skip_n, max_cs_n);

	thr_a1 = async_tensor_get_submatrix(a, blk_ia, cidxa, aidxa,
	    nblk_k, blk_cs_m, skip_k, skip_m, k, NULL, blk_a2);
	thr_a2 = async_tensor_prefetch_next(a, blk_ia, cidxa, aidxa,
	    nblk_k, blk_cs_m, skip_k, skip_m, blk_a4, max_cs_m, 0);
	thr_b1 = async_tensor_get_submatrix(b, blk_ib, cidxb, aidxb,
	    nblk_k, blk_cs_n, skip_k, skip_n, k, NULL, blk_b2);
	thr_b2 = async_tensor_prefetch_next(b, blk_ib, cidxb, aidxb,
	    nblk_k, blk_cs_n, skip_k, skip_n, blk_b4, max_cs_n, split_b);
	thr_c1 = pthread_self();
	thr_c2 = pthread_self();

	done_a = 0;
	while (!done_a) {
		if (pthread_join(thr_a1, NULL))
			xm_log_perror("pthread_join");
		swap_ptr(&blk_a1, &blk_a2);

		if (!pthread_equal(thr_a2, pthread_self()))
			if (pthread_join(thr_a2, NULL))
				xm_log_perror("pthread_join");
		swap_ptr(&blk_a3, &blk_a4);

		done_a = add_chunk_blocks(a, &blk_ia, aidxa,
		    skip_m, max_cs_m);
		if (!done_a) {
			blk_cs_m = get_chunk_blocks(a, blk_ia, aidxa,
			    skip_m, max_cs_m);
			thr_a1 = async_tensor_get_submatrix(a,
			    blk_ia, cidxa, aidxa, nblk_k, blk_cs_m,
			    skip_k, skip_m, k, blk_a3, blk_a2);
			thr_a2 = async_tensor_prefetch_next(a, blk_ia,
			    cidxa, aidxa, nblk_k, blk_cs_m,
			    skip_k, skip_m, blk_a4, max_cs_m, 0);
		}

		blk_ib = xm_dim_zero(b->dim.n);
		done_b = 0;
		while (!done_b) {
			if (!pthread_equal(thr_b1, pthread_self()))
				if (pthread_join(thr_b1, NULL))
					xm_log_perror("pthread_join");
			swap_ptr(&blk_b1, &blk_b2);

			if (!pthread_equal(thr_b2, pthread_self()))
				if (pthread_join(thr_b2, NULL))
					xm_log_perror("pthread_join");
			swap_ptr(&blk_b3, &blk_b4);

			done_b = add_chunk_blocks(b, &blk_ib, aidxb,
			    skip_n, max_cs_n);
			if (split_b && (!done_a || !done_b)) {
				blk_cs_n = get_chunk_blocks(b, blk_ib, aidxb,
				    skip_n, max_cs_n);
				thr_b1 = async_tensor_get_submatrix(b,
				    blk_ib, cidxb, aidxb, nblk_k, blk_cs_n,
				    skip_k, skip_n, k, blk_b3, blk_b2);
				thr_b2 = async_tensor_prefetch_next(b, blk_ib,
				    cidxb, aidxb, nblk_k, blk_cs_n,
				    skip_k, skip_n, blk_b4, max_cs_n,
				    split_b && !done_a);
			} else {
				thr_b1 = pthread_self();
				thr_b2 = pthread_self();
			}

			swap_ptr(&blk_c1, &blk_c2);
			blk_ic2 = blk_ic;

			if (!done_b) {
				add_chunk_blocks(c, &blk_ic, aidxc,
				    skip_n, max_cs_n);
			} else if (!done_a) {
				xm_dim_zero_mask(&blk_ic, &aidxc);
				add_chunk_blocks(c, &blk_ic, cidxc,
				    skip_m, max_cs_m);
			}

			m = get_chunk_size(c, blk_ic2, cidxc, skip_m, max_cs_m);
			n = get_chunk_size(c, blk_ic2, aidxc, skip_n, max_cs_n);

			gemm_timer = timer_start("blas_gemm");
			xm_log("blas_gemm m=%zu n=%zu k=%zu", m, n, k);
			if (aidxc.n > 0 && aidxc.i[0] == 0) {
				gemm_wrapper('T', 'N', (int)n, (int)m, (int)k,
				    alpha, blk_b1, (int)k, blk_a1, (int)k,
				    0.0, blk_c1, (int)max_cs_n);
			} else {
				gemm_wrapper('T', 'N', (int)m, (int)n, (int)k,
				    alpha, blk_a1, (int)k, blk_b1, (int)k,
				    0.0, blk_c1, (int)max_cs_m);
			}
			timer_stop(&gemm_timer);

			if (!pthread_equal(thr_c2, pthread_self())) {
				if (pthread_join(thr_c2, NULL))
					xm_log_perror("pthread_join");
			}
			swap_ptr(&blk_c1, &blk_c3);
			blk_cs_m = get_chunk_blocks(c, blk_ic2, cidxc,
			    skip_m, max_cs_m);
			blk_cs_n = get_chunk_blocks(c, blk_ic2, aidxc,
			    skip_n, max_cs_n);
			if (aidxc.n > 0 && aidxc.i[0] == 0) {
				thr_c2 = async_tensor_set_submatrix(c, blk_ic2,
				    aidxc, cidxc, blk_cs_n, blk_cs_m,
				    skip_n, skip_m, max_cs_n, blk_c3);
			} else {
				thr_c2 = async_tensor_set_submatrix(c, blk_ic2,
				    cidxc, aidxc, blk_cs_m, blk_cs_n,
				    skip_m, skip_n, max_cs_m, blk_c3);
			}
		}
	}
	if (pthread_join(thr_c2, NULL))
		xm_log_perror("pthread_join");
	return (XM_RESULT_SUCCESS);
}

static int
xm_contract_part(xm_scalar_t alpha, struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, xm_dim_t cidxa, xm_dim_t aidxa, xm_dim_t cidxb,
    xm_dim_t aidxb, xm_dim_t cidxc, xm_dim_t aidxc, xm_scalar_t *buf)
{
	/* bit vectors of blocks that can be skipped along m, n, k */
	bitstr_t *skip_m, *skip_n, *skip_k;
	size_t m, n, k;
	int res;

	res = XM_RESULT_SUCCESS;

	m = xm_dim_dot_mask(&c->dim, &cidxc);
	n = xm_dim_dot_mask(&c->dim, &aidxc);
	k = xm_dim_dot_mask(&a->dim, &cidxa);

	skip_m = bit_alloc(m);
	skip_n = bit_alloc(n);
	skip_k = bit_alloc(k);

	bit_nset(skip_m, 0, m - 1);
	make_skip(skip_m, c, cidxc, aidxc);
	set_skip_zero(skip_m, a, aidxa, cidxa);

	bit_nset(skip_n, 0, n - 1);
	make_skip(skip_n, c, aidxc, cidxc);
	set_skip_zero(skip_n, b, aidxb, cidxb);

	bit_nclear(skip_k, 0, k - 1);
	set_skip_zero(skip_k, a, cidxa, aidxa);
	set_skip_zero(skip_k, b, cidxb, aidxb);

	res = xm_do_contract(alpha, a, b, c, cidxa, aidxa,
	    cidxb, aidxb, cidxc, aidxc, skip_m, skip_n, skip_k, buf);

	free(skip_m);
	free(skip_n);
	free(skip_k);
	return (res);
}
#endif

static void
compute_block(struct ctx *ctx, size_t stride, xm_dim_t blkidxc,
    xm_scalar_t *buf)
{
	size_t i, m, n, k = 0;
	xm_scalar_t *buf_a, *buf_b, *blkbuf_a, *blkbuf_b;
	xm_scalar_t *blkbuf_c1, *blkbuf_c2;
	struct xm_block *blk_c = xm_tensor_get_block(ctx->c, &blkidxc);
	xm_dim_t blkidxa, blkidxb;

	m = xm_dim_dot_mask(&blk_c->dim, &ctx->cidxc);
	n = xm_dim_dot_mask(&blk_c->dim, &ctx->aidxc);
	buf_a = buf;
	buf_b = buf_a + stride * m;
	blkbuf_a = buf_b + stride * n;
	blkbuf_b = blkbuf_a + ctx->a->max_block_size;
	blkbuf_c1 = blkbuf_b + ctx->b->max_block_size;
	blkbuf_c2 = blkbuf_c1 + ctx->c->max_block_size;

	xm_scalar_t *buf_a_ptr = buf_a;
	xm_scalar_t *buf_b_ptr = buf_b;
	blkidxa = xm_dim_zero(ctx->a->dim.n);
	blkidxb = xm_dim_zero(ctx->b->dim.n);
	xm_dim_set_mask2(&blkidxa, &ctx->aidxa, &blkidxc, &ctx->cidxc);
	xm_dim_set_mask2(&blkidxb, &ctx->aidxb, &blkidxc, &ctx->aidxc);

	for (i = 0; i < ctx->nblk_k; i++) {
		struct xm_block *blk_a = xm_tensor_get_block(ctx->a, &blkidxa);
		struct xm_block *blk_b = xm_tensor_get_block(ctx->b, &blkidxb);

		if (blk_a->is_nonzero && blk_b->is_nonzero) {
			size_t mblk, nblk, kblk;
			mblk = xm_dim_dot_mask(&blk_a->dim, &ctx->aidxa);
			nblk = xm_dim_dot_mask(&blk_b->dim, &ctx->aidxb);
			kblk = xm_dim_dot_mask(&blk_a->dim, &ctx->cidxa);

			size_t blk_a_size = xm_dim_dot(&blk_a->dim);
			xm_allocator_read(ctx->a->allocator, blk_a->data_ptr,
			    blkbuf_a, blk_a_size * sizeof(xm_scalar_t));
			block_get_matrix(blk_a, ctx->cidxa, ctx->aidxa,
			    kblk, mblk, blkbuf_a, buf_a_ptr, stride);
			buf_a_ptr += kblk;

			size_t blk_b_size = xm_dim_dot(&blk_b->dim);
			xm_allocator_read(ctx->b->allocator, blk_b->data_ptr,
			    blkbuf_b, blk_b_size * sizeof(xm_scalar_t));
			block_get_matrix(blk_b, ctx->cidxb, ctx->aidxb,
			    kblk, nblk, blkbuf_b, buf_b_ptr, stride);
			buf_b_ptr += kblk;

			k += kblk;
		}
		xm_dim_inc_mask(&blkidxa, &ctx->a->dim, &ctx->cidxa);
		xm_dim_inc_mask(&blkidxb, &ctx->b->dim, &ctx->cidxb);
	}

	if (k == 0) {
		size_t size = xm_dim_dot(&blk_c->dim) * sizeof(xm_scalar_t);
		memset(blkbuf_c1, 0, size);
		xm_allocator_write(ctx->c->allocator, blk_c->data_ptr,
		    blkbuf_c1, size);
		return;
	}

	if (ctx->aidxc.n > 0 && ctx->aidxc.i[0] == 0) {
		gemm_wrapper('T', 'N', (int)n, (int)m, (int)k,
		    ctx->alpha, buf_b, (int)stride, buf_a, (int)stride,
		    0.0, blkbuf_c1, (int)n);
		block_set_matrix(blk_c, ctx->aidxc, ctx->cidxc, n, m,
		    blkbuf_c1, n, blkbuf_c2, ctx->c->allocator);
	} else {
		gemm_wrapper('T', 'N', (int)m, (int)n, (int)k,
		    ctx->alpha, buf_a, (int)stride, buf_b, (int)stride,
		    0.0, blkbuf_c1, (int)m);
		block_set_matrix(blk_c, ctx->cidxc, ctx->aidxc, m, n,
		    blkbuf_c1, m, blkbuf_c2, ctx->c->allocator);
	}
}

static size_t
compute_stride_idx(struct ctx *ctx, const xm_dim_t *blkidxc)
{
	struct xm_block *blk_a, *blk_b;
	xm_dim_t blkidxa, blkidxb;
	size_t i, stride = 0;

	blkidxa = xm_dim_zero(ctx->a->dim.n);
	blkidxb = xm_dim_zero(ctx->b->dim.n);
	xm_dim_set_mask2(&blkidxa, &ctx->aidxa, blkidxc, &ctx->cidxc);
	xm_dim_set_mask2(&blkidxb, &ctx->aidxb, blkidxc, &ctx->aidxc);

	for (i = 0; i < ctx->nblk_k; i++) {
		blk_a = xm_tensor_get_block(ctx->a, &blkidxa);
		blk_b = xm_tensor_get_block(ctx->b, &blkidxb);
		if (blk_a->is_nonzero && blk_b->is_nonzero)
			stride += xm_dim_dot_mask(&blk_a->dim, &ctx->cidxa);
		xm_dim_inc_mask(&blkidxa, &ctx->a->dim, &ctx->cidxa);
		xm_dim_inc_mask(&blkidxb, &ctx->b->dim, &ctx->cidxb);
	}
	return (stride);
}

static size_t
compute_stride(struct ctx *ctx)
{
	struct xm_block *blk;
	xm_dim_t blkidxc = xm_dim_zero(ctx->c->dim.n);
	size_t i, j, s, stride = 0;

	for (i = 0; i < ctx->nblk_m; i++) {
		for (j = 0; j < ctx->nblk_n; j++) {
			blk = xm_tensor_get_block(ctx->c, &blkidxc);
			if (blk->is_source) {
				s = compute_stride_idx(ctx, &blkidxc);
				if (s > stride)
					stride = s;
			}
			xm_dim_inc_mask(&blkidxc, &ctx->c->dim, &ctx->aidxc);
		}
		xm_dim_inc_mask(&blkidxc, &ctx->c->dim, &ctx->cidxc);
	}
	return (stride);
}

static size_t
compute_max_mplusn(struct ctx *ctx)
{
	struct xm_block *blk;
	xm_dim_t idx = xm_dim_zero(ctx->c->dim.n);
	size_t i, j, m, n, mplusn = 0;

	for (i = 0; i < ctx->nblk_m; i++) {
		for (j = 0; j < ctx->nblk_n; j++) {
			blk = xm_tensor_get_block(ctx->c, &idx);
			m = xm_dim_dot_mask(&blk->dim, &ctx->cidxc);
			n = xm_dim_dot_mask(&blk->dim, &ctx->aidxc);
			if (m + n > mplusn)
				mplusn = m + n;
			xm_dim_inc_mask(&idx, &ctx->c->dim, &ctx->aidxc);
		}
		xm_dim_inc_mask(&idx, &ctx->c->dim, &ctx->cidxc);
	}
	return (mplusn);
}

static xm_dim_t *
get_nonzero_blocks(struct ctx *ctx, size_t *nnzblkout)
{
	struct xm_block *blk;
	xm_dim_t *nzblk, *nzblkptr, idx;
	size_t i, j, nnzblk = 0;

	idx = xm_dim_zero(ctx->c->dim.n);
	for (i = 0; i < ctx->nblk_m; i++) {
		for (j = 0; j < ctx->nblk_n; j++) {
			blk = xm_tensor_get_block(ctx->c, &idx);
			if (blk->is_source)
				nnzblk++;
			xm_dim_inc_mask(&idx, &ctx->c->dim, &ctx->aidxc);
		}
		xm_dim_inc_mask(&idx, &ctx->c->dim, &ctx->cidxc);
	}

	nzblkptr = nzblk = xmalloc(nnzblk * sizeof(xm_dim_t));

	idx = xm_dim_zero(ctx->c->dim.n);
	for (i = 0; i < ctx->nblk_m; i++) {
		for (j = 0; j < ctx->nblk_n; j++) {
			blk = xm_tensor_get_block(ctx->c, &idx);
			if (blk->is_source)
				*nzblkptr++ = idx;
			xm_dim_inc_mask(&idx, &ctx->c->dim, &ctx->aidxc);
		}
		xm_dim_inc_mask(&idx, &ctx->c->dim, &ctx->cidxc);
	}

	*nnzblkout = nnzblk;
	return (nzblk);
}

void
xm_contract(xm_scalar_t alpha, struct xm_tensor *a, struct xm_tensor *b,
    struct xm_tensor *c, const char *idxa, const char *idxb, const char *idxc)
{
	struct ctx ctx;
	struct timer timer;
	xm_dim_t cidxa, aidxa, cidxb, aidxb, cidxc, aidxc, *nzblk;
	size_t i, si1, si2, stride, max_mplusn, sz, nnzblk;
	int sym_k;

	assert(xm_tensor_is_initialized(a));
	assert(xm_tensor_is_initialized(b));
	assert(xm_tensor_is_initialized(c));
	assert(strlen(idxa) == a->dim.n);
	assert(strlen(idxb) == b->dim.n);
	assert(strlen(idxc) == c->dim.n);

	parse_idx(idxa, idxb, &cidxa, &cidxb);
	parse_idx(idxc, idxa, &cidxc, &aidxa);
	parse_idx(idxc, idxb, &aidxc, &aidxb);

	ctx.alpha = alpha;
	ctx.a = a, ctx.b = b, ctx.c = c;
	ctx.cidxa = cidxa, ctx.aidxa = aidxa;
	ctx.cidxb = cidxb, ctx.aidxb = aidxb;
	ctx.cidxc = cidxc, ctx.aidxc = aidxc;
	ctx.nblk_m = xm_dim_dot_mask(&a->dim, &aidxa);
	ctx.nblk_n = xm_dim_dot_mask(&b->dim, &aidxb);
	ctx.nblk_k = xm_dim_dot_mask(&a->dim, &cidxa);

	assert(aidxa.n + cidxa.n == a->dim.n);
	assert(aidxb.n + cidxb.n == b->dim.n);
	assert(aidxc.n + cidxc.n == c->dim.n);
	assert((aidxc.n > 0 && aidxc.i[0] == 0) ||
	       (cidxc.n > 0 && cidxc.i[0] == 0));

	check_block_consistency(a, b, cidxa, cidxb);
	check_block_consistency(c, a, cidxc, aidxa);
	check_block_consistency(c, b, aidxc, aidxb);

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

	if (sym_k) {
		xm_log("enabling k symmetry");
		set_k_symmetry(a, cidxa, aidxa, si1, si2, 1);
	}

	nzblk = get_nonzero_blocks(&ctx, &nnzblk);
	stride = compute_stride(&ctx);
	max_mplusn = compute_max_mplusn(&ctx);
	sz = stride * max_mplusn + a->max_block_size +
	    b->max_block_size + 2 * c->max_block_size;
	timer = timer_start("xm_contract");
#pragma omp parallel private(i)
{
	xm_scalar_t *buf = xmalloc(sz * sizeof(xm_scalar_t));
#pragma omp for schedule(dynamic)
	for (i = 0; i < nnzblk; i++)
		compute_block(&ctx, stride, nzblk[i], buf);
	free(buf);
}
	timer_stop(&timer);

	if (sym_k) {
		xm_log("disabling k symmetry");
		set_k_symmetry(a, cidxa, aidxa, si1, si2, 0);
	}
	free(nzblk);
}
