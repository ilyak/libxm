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
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*unfold_test_fn)(const char *);
typedef void (*make_abc_fn)(xm_allocator_t *, xm_tensor_t **, xm_tensor_t **,
    xm_tensor_t **);

struct contract_test {
	make_abc_fn make_abc;
	const char *idxa, *idxb, *idxc;
};

static void
fatal(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fprintf(stderr, "\n");
	abort();
}

static int
scalar_eq(xm_scalar_t a, xm_scalar_t b)
{
#if defined(XM_SCALAR_FLOAT)
	return fabsf(a - b) < 1.0e-4f;
#elif defined(XM_SCALAR_DOUBLE_COMPLEX)
	return cabs(a - b) < 1.0e-8;
#elif defined(XM_SCALAR_FLOAT_COMPLEX)
	return cabsf(a - b) < 1.0e-4f;
#else /* assume double */
	return fabs(a - b) < 1.0e-8;
#endif
}

static xm_scalar_t
random_scalar(void)
{
#if defined(XM_SCALAR_DOUBLE_COMPLEX) || defined(XM_SCALAR_FLOAT_COMPLEX)
	return (xm_scalar_t)((drand48() - 0.5) + (drand48() - 0.5) * I);
#else
	return (xm_scalar_t)(drand48() - 0.5);
#endif
}

static void
fill_random(xm_tensor_t *t)
{
	xm_allocator_t *allocator;
	const xm_block_space_t *bs;
	xm_dim_t idx, nblocks;
	xm_scalar_t *data;
	size_t i, blksize, maxblksize;
	uintptr_t ptr;
	int type;

	allocator = xm_tensor_get_allocator(t);
	bs = xm_tensor_get_block_space(t);
	maxblksize = xm_block_space_get_largest_block_size(bs);
	data = malloc(maxblksize * sizeof(xm_scalar_t));
	assert(data);
	nblocks = xm_block_space_get_nblocks(bs);
	idx = xm_dim_zero(nblocks.n);
	while (xm_dim_ne(&idx, &nblocks)) {
		type = xm_tensor_get_block_type(t, idx);
		if (type == XM_BLOCK_TYPE_CANONICAL) {
			blksize = xm_tensor_get_block_size(t, idx);
			for (i = 0; i < blksize; i++)
				data[i] = random_scalar();
			ptr = xm_tensor_get_block_data_ptr(t, idx);
			xm_allocator_write(allocator, ptr, data,
			    blksize * sizeof(xm_scalar_t));
		}
		xm_dim_inc(&idx, &nblocks);
	}
	free(data);
}

static void
compare_tensors(xm_tensor_t *t, xm_tensor_t *u)
{
	xm_dim_t idx, dimst, dimsu;

	dimst = xm_tensor_get_abs_dims(t);
	dimsu = xm_tensor_get_abs_dims(u);
	assert(xm_dim_eq(&dimst, &dimsu));

	idx = xm_dim_zero(dimst.n);
	while (xm_dim_ne(&idx, &dimst)) {
		xm_scalar_t et = xm_tensor_get_element(t, idx);
		xm_scalar_t eu = xm_tensor_get_element(u, idx);
		if (!scalar_eq(et, eu))
			fatal("%s: tensors do not match", __func__);
		xm_dim_inc(&idx, &dimst);
	}
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

static void
check_result(xm_tensor_t *cc, xm_scalar_t alpha, xm_tensor_t *a, xm_tensor_t *b,
    xm_scalar_t beta, xm_tensor_t *c, const char *idxa, const char *idxb,
    const char *idxc)
{
	xm_dim_t absdimsa, absdimsb, absdimsc, ia, ib, ic;
	xm_dim_t cidxa, aidxa, cidxb, aidxb, cidxc, aidxc;
	xm_scalar_t ref, ecc;
	size_t k, nk;

	make_masks(idxa, idxb, &cidxa, &cidxb);
	make_masks(idxc, idxa, &cidxc, &aidxa);
	make_masks(idxc, idxb, &aidxc, &aidxb);
	absdimsa = xm_tensor_get_abs_dims(a);
	absdimsb = xm_tensor_get_abs_dims(b);
	absdimsc = xm_tensor_get_abs_dims(c);
	nk = xm_dim_dot_mask(&absdimsa, &cidxa);
	ic = xm_dim_zero(absdimsc.n);
	while (xm_dim_ne(&ic, &absdimsc)) {
		ref = beta * xm_tensor_get_element(c, ic);
		ia = xm_dim_zero(absdimsa.n);
		ib = xm_dim_zero(absdimsb.n);
		xm_dim_set_mask(&ia, &aidxa, &ic, &cidxc);
		xm_dim_set_mask(&ib, &aidxb, &ic, &aidxc);
		for (k = 0; k < nk; k++) {
			xm_scalar_t ea = xm_tensor_get_element(a, ia);
			xm_scalar_t eb = xm_tensor_get_element(b, ib);
			ref += alpha * ea * eb;
			xm_dim_inc_mask(&ia, &absdimsa, &cidxa);
			xm_dim_inc_mask(&ib, &absdimsb, &cidxb);
		}
		ecc = xm_tensor_get_element(cc, ic);
		if (!scalar_eq(ecc, ref))
			fatal("%s: result != ref", __func__);
		xm_dim_inc(&ic, &absdimsc);
	}
}

static void
test_contract(struct contract_test *test, const char *path, xm_scalar_t alpha,
    xm_scalar_t beta)
{
	xm_allocator_t *allocator;
	xm_tensor_t *a, *b, *c, *cc;

	allocator = xm_allocator_create(path);
	assert(allocator);
	test->make_abc(allocator, &a, &b, &c);
	assert(a);
	assert(b);
	assert(c);
	fill_random(a);
	fill_random(b);
	fill_random(c);
	cc = xm_tensor_clone(c, allocator);
	assert(cc);
	xm_contract(alpha, a, b, beta, cc, test->idxa, test->idxb, test->idxc);
	check_result(cc, alpha, a, b, beta, c, test->idxa, test->idxb,
	    test->idxc);
	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free_block_data(c);
	xm_tensor_free_block_data(cc);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_tensor_free(c);
	xm_tensor_free(cc);
	xm_allocator_destroy(allocator);
}

static void
make_abc_1(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx;
	xm_block_space_t *bsa;
	xm_tensor_t *a, *b, *c;
	uintptr_t ptr;

	bsa = xm_block_space_create(xm_dim_2(4, 4));
	a = xm_tensor_create(bsa, allocator);
	b = xm_tensor_create(bsa, allocator);
	c = xm_tensor_create(bsa, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_2(0, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_2(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx, nblocks;
	xm_block_space_t *bsa;
	xm_tensor_t *a, *b, *c;
	uintptr_t ptr;

	bsa = xm_block_space_create(xm_dim_2(4, 4));
	xm_block_space_split(bsa, 0, 2);
	xm_block_space_split(bsa, 1, 2);
	a = xm_tensor_create(bsa, allocator);
	b = xm_tensor_create(bsa, allocator);
	c = xm_tensor_create(bsa, allocator);
	xm_block_space_free(bsa);
	nblocks = xm_tensor_get_nblocks(a);
	for (idx = xm_dim_zero(nblocks.n);
	     xm_dim_ne(&idx, &nblocks);
	     xm_dim_inc(&idx, &nblocks)) {
		if (xm_tensor_get_block_type(a, idx) != XM_BLOCK_TYPE_ZERO)
			fatal("%s: unexpected block type", __func__);
		ptr = xm_tensor_allocate_block_data(a, idx);
		xm_tensor_set_canonical_block(a, idx, ptr);
		ptr = xm_tensor_allocate_block_data(b, idx);
		xm_tensor_set_canonical_block(b, idx, ptr);
		ptr = xm_tensor_allocate_block_data(c, idx);
		xm_tensor_set_canonical_block(c, idx, ptr);
	}
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_3(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx;
	xm_block_space_t *bsa, *bsc;
	xm_tensor_t *a, *b, *c;
	uintptr_t ptr;

	bsa = xm_block_space_create(xm_dim_same(8, 1));
	a = xm_tensor_create(bsa, allocator);
	b = xm_tensor_create(bsa, allocator);
	xm_block_space_free(bsa);
	bsc = xm_block_space_create(xm_dim_same(2, 1));
	c = xm_tensor_create(bsc, allocator);
	xm_block_space_free(bsc);
	idx = xm_dim_zero(8);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);
	idx = xm_dim_zero(2);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_4(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx;
	xm_block_space_t *bsa;
	xm_tensor_t *a, *b, *c;
	uintptr_t ptr;

	bsa = xm_block_space_create(xm_dim_same(6, 3));
	a = xm_tensor_create(bsa, allocator);
	b = xm_tensor_create(bsa, allocator);
	c = xm_tensor_create(bsa, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_zero(6);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_5(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx;
	xm_block_space_t *bsa;
	xm_tensor_t *a, *b, *c;
	uintptr_t ptr;

	bsa = xm_block_space_create(xm_dim_2(4, 4));
	xm_block_space_split(bsa, 0, 2);
	xm_block_space_split(bsa, 1, 2);
	a = xm_tensor_create(bsa, allocator);
	b = xm_tensor_create(bsa, allocator);
	c = xm_tensor_create(bsa, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_2(0, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	idx = xm_dim_2(0, 1);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	idx = xm_dim_2(1, 0);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	idx = xm_dim_2(1, 1);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_6(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx;
	xm_block_space_t *bsa, *bsb, *bsc;
	xm_tensor_t *a, *b, *c;
	uintptr_t ptr;

	bsa = xm_block_space_create(xm_dim_1(3));
	xm_block_space_split(bsa, 0, 1);
	a = xm_tensor_create(bsa, allocator);
	xm_block_space_free(bsa);
	bsb = xm_block_space_create(xm_dim_2(2, 5));
	xm_block_space_split(bsb, 1, 3);
	b = xm_tensor_create(bsb, allocator);
	xm_block_space_free(bsb);
	bsc = xm_block_space_create(xm_dim_3(2, 3, 5));
	xm_block_space_split(bsc, 1, 1);
	xm_block_space_split(bsc, 2, 3);
	c = xm_tensor_create(bsc, allocator);
	xm_block_space_free(bsc);

	idx = xm_dim_1(0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	idx = xm_dim_1(1);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);

	idx = xm_dim_2(0, 0);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);
	idx = xm_dim_2(0, 1);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);

	idx = xm_dim_3(0, 0, 0);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	idx = xm_dim_3(0, 0, 1);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	idx = xm_dim_3(0, 1, 0);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	idx = xm_dim_3(0, 1, 1);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_7(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx, nblocks;
	xm_block_space_t *bsa, *bsb, *bsc;
	xm_tensor_t *a, *b, *c;
	uintptr_t ptr;

	bsa = xm_block_space_create(xm_dim_3(7, 4, 11));
	xm_block_space_split(bsa, 0, 2);
	xm_block_space_split(bsa, 0, 5);
	xm_block_space_split(bsa, 1, 2);
	xm_block_space_split(bsa, 2, 2);
	xm_block_space_split(bsa, 2, 9);
	a = xm_tensor_create(bsa, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_3(0, 0, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	idx = xm_dim_3(1, 0, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	idx = xm_dim_3(2, 0, 0);
	xm_tensor_set_derivative_block(a, idx, xm_dim_3(0, 0, 0),
	    xm_dim_3(2, 1, 0), -0.5);
	idx = xm_dim_3(0, 1, 0);
	xm_tensor_set_derivative_block(a, idx, xm_dim_3(0, 0, 0),
	    xm_dim_3(1, 0, 2), 1.5);
	idx = xm_dim_3(1, 1, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	idx = xm_dim_3(2, 1, 0);
	xm_tensor_set_derivative_block(a, idx, xm_dim_3(0, 0, 0),
	    xm_dim_3(2, 0, 1), 0.7);
	nblocks = xm_tensor_get_nblocks(a);
	for (idx = xm_dim_zero(nblocks.n);
	     xm_dim_ne(&idx, &nblocks);
	     xm_dim_inc(&idx, &nblocks)) {
		if (xm_tensor_get_block_type(a, idx) == XM_BLOCK_TYPE_ZERO) {
			xm_dim_t tt[] = { xm_dim_3(0, 0, 0),
					  xm_dim_3(2, 1, 0) };
			if (xm_dim_eq(&idx, &tt[0]) ||
			    xm_dim_eq(&idx, &tt[1]))
				fatal("%s: unexpected block type", __func__);
			ptr = xm_tensor_allocate_block_data(a, idx);
			xm_tensor_set_canonical_block(a, idx, ptr);
		}
	}

	bsb = xm_block_space_create(xm_dim_3(7, 4, 8));
	xm_block_space_split(bsb, 0, 2);
	xm_block_space_split(bsb, 0, 5);
	xm_block_space_split(bsb, 1, 2);
	xm_block_space_split(bsb, 2, 2);
	xm_block_space_split(bsb, 2, 6);
	b = xm_tensor_create(bsb, allocator);
	xm_block_space_free(bsb);
	nblocks = xm_tensor_get_nblocks(b);
	for (idx = xm_dim_zero(nblocks.n);
	     xm_dim_ne(&idx, &nblocks);
	     xm_dim_inc(&idx, &nblocks)) {
		ptr = xm_tensor_allocate_block_data(b, idx);
		xm_tensor_set_canonical_block(b, idx, ptr);
	}

	bsc = xm_block_space_create(xm_dim_2(8, 11));
	xm_block_space_split(bsc, 0, 2);
	xm_block_space_split(bsc, 0, 6);
	xm_block_space_split(bsc, 1, 2);
	xm_block_space_split(bsc, 1, 9);
	c = xm_tensor_create(bsc, allocator);
	xm_block_space_free(bsc);
	nblocks = xm_tensor_get_nblocks(c);
	for (idx = xm_dim_zero(nblocks.n);
	     xm_dim_ne(&idx, &nblocks);
	     xm_dim_inc(&idx, &nblocks)) {
		ptr = xm_tensor_allocate_block_data(c, idx);
		xm_tensor_set_canonical_block(c, idx, ptr);
	}
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_8(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx;
	xm_block_space_t *bsa, *bsb, *bsc;
	xm_tensor_t *a, *b, *c;
	uintptr_t ptr;

	bsa = xm_block_space_create(xm_dim_2(15, 10));
	xm_block_space_split(bsa, 0, 5);
	xm_block_space_split(bsa, 0, 10);
	xm_block_space_split(bsa, 1, 5);
	a = xm_tensor_create(bsa, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_2(0, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	idx = xm_dim_2(0, 1);
	xm_tensor_set_derivative_block(a, idx, xm_dim_2(0, 0),
	    xm_dim_2(1, 0), -0.3);
	idx = xm_dim_2(2, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	idx = xm_dim_2(2, 1);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);

	bsb = xm_block_space_create(xm_dim_1(10));
	xm_block_space_split(bsb, 0, 5);
	b = xm_tensor_create(bsb, allocator);
	xm_block_space_free(bsb);
	idx = xm_dim_1(0);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);
	idx = xm_dim_1(1);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);

	bsc = xm_block_space_create(xm_dim_1(15));
	xm_block_space_split(bsc, 0, 5);
	xm_block_space_split(bsc, 0, 10);
	c = xm_tensor_create(bsc, allocator);
	xm_block_space_free(bsc);
	idx = xm_dim_1(0);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);
	/* idx = xm_dim_1(1) stays zero */
	idx = xm_dim_1(2);
	ptr = xm_tensor_allocate_block_data(c, idx);
	xm_tensor_set_canonical_block(c, idx, ptr);

	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_9(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx, idx2, perm;
	xm_block_space_t *bsa, *bsb;
	xm_tensor_t *a, *b, *c;
	uintptr_t ptr;
	const size_t o = 8, v = 11, blocksize = 3;
	const size_t nblko = 3, nblkv = 4;
	size_t i;

	bsa = xm_block_space_create(xm_dim_4(o, o, v, v));
	for (i = 1; i < nblko; i++) {
		xm_block_space_split(bsa, 0, i * blocksize);
		xm_block_space_split(bsa, 1, i * blocksize);
	}
	for (i = 1; i < nblkv; i++) {
		xm_block_space_split(bsa, 2, i * blocksize);
		xm_block_space_split(bsa, 3, i * blocksize);
	}
	a = xm_tensor_create(bsa, allocator);
	c = xm_tensor_create(bsa, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_zero(4);
	for (idx.i[0] = 0; idx.i[0] < nblko; idx.i[0]++)
	for (idx.i[1] = 0; idx.i[1] < nblko; idx.i[1]++)
	for (idx.i[2] = 0; idx.i[2] < nblkv; idx.i[2]++)
	for (idx.i[3] = 0; idx.i[3] < nblkv; idx.i[3]++) {
		if (xm_tensor_get_block_type(a, idx) != XM_BLOCK_TYPE_ZERO)
			continue;
		ptr = xm_tensor_allocate_block_data(a, idx);
		xm_tensor_set_canonical_block(a, idx, ptr);
		ptr = xm_tensor_allocate_block_data(c, idx);
		xm_tensor_set_canonical_block(c, idx, ptr);

		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[2], idx.i[3]);
		perm = xm_dim_4(1, 0, 2, 3);
		if (xm_tensor_get_block_type(a, idx) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[0], idx.i[1], idx.i[3], idx.i[2]);
		perm = xm_dim_4(0, 1, 3, 2);
		if (xm_tensor_get_block_type(a, idx) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[3], idx.i[2]);
		perm = xm_dim_4(1, 0, 3, 2);
		if (xm_tensor_get_block_type(a, idx) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, 1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, 1);
		}
	}

	bsb = xm_block_space_create(xm_dim_4(v, v, v, v));
	for (i = 1; i < nblkv; i++) {
		xm_block_space_split(bsb, 0, i * blocksize);
		xm_block_space_split(bsb, 1, i * blocksize);
		xm_block_space_split(bsb, 2, i * blocksize);
		xm_block_space_split(bsb, 3, i * blocksize);
	}
	b = xm_tensor_create(bsb, allocator);
	xm_block_space_free(bsb);
	idx = xm_dim_zero(4);
	for (idx.i[0] = 0; idx.i[0] < nblkv; idx.i[0]++)
	for (idx.i[1] = 0; idx.i[1] < nblkv; idx.i[1]++)
	for (idx.i[2] = 0; idx.i[2] < nblkv; idx.i[2]++)
	for (idx.i[3] = 0; idx.i[3] < nblkv; idx.i[3]++) {
		if (xm_tensor_get_block_type(b, idx) != XM_BLOCK_TYPE_ZERO)
			continue;
		ptr = xm_tensor_allocate_block_data(b, idx);
		xm_tensor_set_canonical_block(b, idx, ptr);

		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[2], idx.i[3]);
		perm = xm_dim_4(1, 0, 2, 3);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[0], idx.i[1], idx.i[3], idx.i[2]);
		perm = xm_dim_4(0, 1, 3, 2);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[2], idx.i[3], idx.i[0], idx.i[1]);
		perm = xm_dim_4(2, 3, 0, 1);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
		idx2 = xm_dim_4(idx.i[3], idx.i[2], idx.i[0], idx.i[1]);
		perm = xm_dim_4(3, 2, 0, 1);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[2], idx.i[3], idx.i[1], idx.i[0]);
		perm = xm_dim_4(2, 3, 1, 0);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[3], idx.i[2], idx.i[1], idx.i[0]);
		perm = xm_dim_4(3, 2, 1, 0);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
	}
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_10(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx, idx2, perm;
	xm_block_space_t *bsa, *bsb;
	xm_tensor_t *a, *b, *c;
	uintptr_t ptr;
	const size_t o = 6, v = 9;
	const size_t nblko = 2, nblkv = 3;

	bsa = xm_block_space_create(xm_dim_4(o, o, v, v));
	xm_block_space_split(bsa, 0, 4);
	xm_block_space_split(bsa, 1, 4);
	xm_block_space_split(bsa, 2, 3);
	xm_block_space_split(bsa, 3, 3);
	xm_block_space_split(bsa, 2, 7);
	xm_block_space_split(bsa, 3, 7);
	a = xm_tensor_create(bsa, allocator);
	c = xm_tensor_create(bsa, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_zero(4);
	for (idx.i[0] = 0; idx.i[0] < nblko; idx.i[0]++)
	for (idx.i[1] = 0; idx.i[1] < nblko; idx.i[1]++)
	for (idx.i[2] = 0; idx.i[2] < nblkv; idx.i[2]++)
	for (idx.i[3] = 0; idx.i[3] < nblkv; idx.i[3]++) {
		if (xm_tensor_get_block_type(a, idx) != XM_BLOCK_TYPE_ZERO)
			continue;
		ptr = xm_tensor_allocate_block_data(a, idx);
		xm_tensor_set_canonical_block(a, idx, ptr);
		ptr = xm_tensor_allocate_block_data(c, idx);
		xm_tensor_set_canonical_block(c, idx, ptr);

		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[2], idx.i[3]);
		perm = xm_dim_4(1, 0, 2, 3);
		if (xm_tensor_get_block_type(a, idx) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[0], idx.i[1], idx.i[3], idx.i[2]);
		perm = xm_dim_4(0, 1, 3, 2);
		if (xm_tensor_get_block_type(a, idx) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, -1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, -1);
		}
		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[3], idx.i[2]);
		perm = xm_dim_4(1, 0, 3, 2);
		if (xm_tensor_get_block_type(a, idx) == XM_BLOCK_TYPE_ZERO) {
			xm_tensor_set_derivative_block(a, idx2, idx, perm, 1);
			xm_tensor_set_derivative_block(c, idx2, idx, perm, 1);
		}
	}

	bsb = xm_block_space_create(xm_dim_4(v, v, v, v));
	xm_block_space_split(bsb, 0, 3);
	xm_block_space_split(bsb, 1, 3);
	xm_block_space_split(bsb, 2, 3);
	xm_block_space_split(bsb, 3, 3);
	xm_block_space_split(bsb, 0, 7);
	xm_block_space_split(bsb, 1, 7);
	xm_block_space_split(bsb, 2, 7);
	xm_block_space_split(bsb, 3, 7);
	b = xm_tensor_create(bsb, allocator);
	xm_block_space_free(bsb);
	idx = xm_dim_zero(4);
	for (idx.i[0] = 0; idx.i[0] < nblkv; idx.i[0]++)
	for (idx.i[1] = 0; idx.i[1] < nblkv; idx.i[1]++)
	for (idx.i[2] = 0; idx.i[2] < nblkv; idx.i[2]++)
	for (idx.i[3] = 0; idx.i[3] < nblkv; idx.i[3]++) {
		if (xm_tensor_get_block_type(b, idx) != XM_BLOCK_TYPE_ZERO)
			continue;
		ptr = xm_tensor_allocate_block_data(b, idx);
		xm_tensor_set_canonical_block(b, idx, ptr);

		idx2 = xm_dim_4(idx.i[1], idx.i[0], idx.i[2], idx.i[3]);
		perm = xm_dim_4(1, 0, 2, 3);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[0], idx.i[1], idx.i[3], idx.i[2]);
		perm = xm_dim_4(0, 1, 3, 2);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[2], idx.i[3], idx.i[0], idx.i[1]);
		perm = xm_dim_4(2, 3, 0, 1);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
		idx2 = xm_dim_4(idx.i[3], idx.i[2], idx.i[0], idx.i[1]);
		perm = xm_dim_4(3, 2, 0, 1);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[2], idx.i[3], idx.i[1], idx.i[0]);
		perm = xm_dim_4(2, 3, 1, 0);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, -1);
		idx2 = xm_dim_4(idx.i[3], idx.i[2], idx.i[1], idx.i[0]);
		perm = xm_dim_4(3, 2, 1, 0);
		if (xm_tensor_get_block_type(b, idx) == XM_BLOCK_TYPE_ZERO)
			xm_tensor_set_derivative_block(b, idx2, idx, perm, 1);
	}
	*aa = b; /* swap */
	*bb = a; /* swap */
	*cc = c;
}

static void
make_abc_11(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx, idx2, perm;
	xm_block_space_t *bsa, *bsb;
	xm_tensor_t *a, *b;
	uintptr_t ptr;

	bsa = xm_block_space_create(xm_dim_4(6, 6, 6, 6));
	xm_block_space_split(bsa, 0, 2);
	xm_block_space_split(bsa, 1, 2);
	xm_block_space_split(bsa, 2, 2);
	xm_block_space_split(bsa, 3, 2);
	a = xm_tensor_create(bsa, allocator);
	xm_block_space_free(bsa);
	idx = xm_dim_4(0, 0, 0, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	idx = xm_dim_4(1, 0, 0, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	idx = xm_dim_4(0, 1, 0, 0);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(1, 0, 2, 3);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(1, 1, 0, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	idx = xm_dim_4(0, 0, 1, 0);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, 1.0);
	idx = xm_dim_4(1, 0, 1, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	idx = xm_dim_4(0, 1, 1, 0);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(1, 0, 2, 3);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(1, 1, 1, 0);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);
	idx = xm_dim_4(0, 0, 0, 1);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(2, 3, 1, 0);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(1, 0, 0, 1);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(0, 1, 3, 2);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(0, 1, 0, 1);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(1, 0, 3, 2);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, 1.0);
	idx = xm_dim_4(1, 1, 0, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(0, 1, 3, 2);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(0, 0, 1, 1);
	idx2 = xm_dim_4(1, 1, 0, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, 1.0);
	idx = xm_dim_4(1, 0, 1, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, 1.0);
	idx = xm_dim_4(0, 1, 1, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(3, 2, 0, 1);
	xm_tensor_set_derivative_block(a, idx, idx2, perm, -1.0);
	idx = xm_dim_4(1, 1, 1, 1);
	ptr = xm_tensor_allocate_block_data(a, idx);
	xm_tensor_set_canonical_block(a, idx, ptr);

	bsb = xm_block_space_create(xm_dim_4(3, 3, 6, 6));
	xm_block_space_split(bsb, 2, 2);
	xm_block_space_split(bsb, 3, 2);
	b = xm_tensor_create(bsb, allocator);
	xm_block_space_free(bsb);
	idx = xm_dim_4(0, 0, 0, 0);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);
	idx = xm_dim_4(0, 0, 0, 1);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);
	idx = xm_dim_4(0, 0, 1, 0);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);
	idx = xm_dim_4(0, 0, 1, 1);
	ptr = xm_tensor_allocate_block_data(b, idx);
	xm_tensor_set_canonical_block(b, idx, ptr);

	*aa = a;
	*bb = b;
	*cc = xm_tensor_clone(b, NULL);
}

static struct contract_test contract_tests[] = {
	{ make_abc_1, "ik", "kj", "ij" },
	{ make_abc_1, "ik", "kj", "ji" },
	{ make_abc_1, "ik", "jk", "ij" },
	{ make_abc_1, "ik", "jk", "ji" },
	{ make_abc_1, "ki", "kj", "ij" },
	{ make_abc_1, "ki", "kj", "ji" },
	{ make_abc_1, "ki", "jk", "ij" },
	{ make_abc_1, "ki", "jk", "ji" },
	{ make_abc_2, "ik", "kj", "ij" },
	{ make_abc_2, "ik", "kj", "ji" },
	{ make_abc_2, "ik", "jk", "ij" },
	{ make_abc_2, "ik", "jk", "ji" },
	{ make_abc_2, "ki", "kj", "ij" },
	{ make_abc_2, "ki", "kj", "ji" },
	{ make_abc_2, "ki", "jk", "ij" },
	{ make_abc_2, "ki", "jk", "ji" },
	{ make_abc_3, "abcdefgh", "abcdefgi", "ih" },
	{ make_abc_3, "abcdefgi", "abcdefgh", "ih" },
	{ make_abc_3, "abcdxfgh", "abcdyfgh", "xy" },
	{ make_abc_3, "abcdefgh", "obcdefgh", "ao" },
	{ make_abc_3, "abcdefgh", "obcdefgh", "oa" },
	{ make_abc_4, "abcdef", "abcijk", "ijkdef" },
	{ make_abc_4, "abcdef", "aibjck", "ijkdef" },
	{ make_abc_4, "badcfe", "xyzdef", "xyzabc" },
	{ make_abc_5, "ik", "kj", "ij" },
	{ make_abc_5, "ik", "kj", "ji" },
	{ make_abc_5, "ik", "jk", "ij" },
	{ make_abc_5, "ik", "jk", "ji" },
	{ make_abc_5, "ki", "kj", "ij" },
	{ make_abc_5, "ki", "kj", "ji" },
	{ make_abc_5, "ki", "jk", "ij" },
	{ make_abc_5, "ki", "jk", "ji" },
	{ make_abc_6, "y", "xz", "xyz" },
	{ make_abc_7, "abc", "abd", "dc" },
	{ make_abc_8, "ab", "b", "a" },
	{ make_abc_9, "ijab", "abcd", "ijcd" },
	{ make_abc_9, "ijcd", "abcd", "ijab" },
	{ make_abc_10, "abcd", "ijab", "ijcd" },
	{ make_abc_10, "cdab", "ijab", "ijcd" },
	{ make_abc_11, "abcd", "ijcd", "ijab" },
};

static void
unfold_test_1(const char *path)
{
	xm_allocator_t *allocator_t, *allocator_u;
	xm_block_space_t *bs;
	xm_tensor_t *t, *u;
	xm_scalar_t *buf1, *buf2;
	uintptr_t ptr;
	size_t i;

	allocator_t = xm_allocator_create(path);
	assert(allocator_t);
	bs = xm_block_space_create(xm_dim_1(15));
	assert(bs);
	xm_block_space_split(bs, 0, 5);
	xm_block_space_split(bs, 0, 10);
	t = xm_tensor_create(bs, allocator_t);
	assert(t);
	xm_block_space_free(bs);
	bs = NULL;
	ptr = xm_tensor_allocate_block_data(t, xm_dim_1(0));
	xm_tensor_set_canonical_block(t, xm_dim_1(0), ptr);
	xm_tensor_set_derivative_block(t, xm_dim_1(1), xm_dim_1(0),
	    xm_dim_identity_permutation(1), 0.5);
	fill_random(t);
	allocator_u = xm_allocator_create(NULL);
	assert(allocator_u);
	u = xm_tensor_clone(t, allocator_u);
	assert(u);
	buf1 = malloc(5 * sizeof(xm_scalar_t));
	assert(buf1);
	buf2 = malloc(5 * sizeof(xm_scalar_t));
	assert(buf2);
	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_1(1));
	xm_allocator_read(allocator_t, ptr, buf1, 5 * sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_1(1), xm_dim_1(0), xm_dim_zero(0),
	    buf1, buf2, 5);
	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_1(0));
	xm_allocator_read(allocator_t, ptr, buf1, 5 * sizeof(xm_scalar_t));
	for (i = 0; i < 5; i++)
		if (!scalar_eq(0.5*buf1[i], buf2[i]))
			fatal("%s: comparison failed", __func__);
	xm_tensor_unfold_block(t, xm_dim_1(0), xm_dim_1(0), xm_dim_zero(0),
	    buf1, buf2, 5);
	xm_tensor_fold_block(t, xm_dim_1(0), xm_dim_1(0), xm_dim_zero(0),
	    buf2, buf1, 5);
	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_1(0));
	xm_allocator_write(allocator_t, ptr, buf1, 5 * sizeof(xm_scalar_t));
	compare_tensors(t, u);
	free(buf1);
	free(buf2);
	xm_tensor_free_block_data(t);
	xm_tensor_free_block_data(u);
	xm_tensor_free(t);
	xm_tensor_free(u);
	xm_allocator_destroy(allocator_t);
	xm_allocator_destroy(allocator_u);
}

static void
unfold_test_2(const char *path)
{
	xm_allocator_t *allocator_t, *allocator_u;
	xm_block_space_t *bs;
	xm_tensor_t *t, *u;
	xm_scalar_t *buf1, *buf2;
	uintptr_t ptr;
	size_t i;

	allocator_t = xm_allocator_create(path);
	assert(allocator_t);
	bs = xm_block_space_create(xm_dim_2(5, 10));
	assert(bs);
	xm_block_space_split(bs, 1, 5);
	t = xm_tensor_create(bs, allocator_t);
	assert(t);
	xm_block_space_free(bs);
	bs = NULL;
	ptr = xm_tensor_allocate_block_data(t, xm_dim_2(0, 0));
	xm_tensor_set_canonical_block(t, xm_dim_2(0, 0), ptr);
	xm_tensor_set_derivative_block(t, xm_dim_2(0, 1), xm_dim_2(0, 0),
	    xm_dim_2(1, 0), -0.3);
	fill_random(t);
	allocator_u = xm_allocator_create(NULL);
	assert(allocator_u);
	u = xm_tensor_clone(t, allocator_u);
	assert(u);
	buf1 = malloc(25 * sizeof(xm_scalar_t));
	assert(buf1);
	buf2 = malloc(25 * sizeof(xm_scalar_t));
	assert(buf2);
	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 1));
	xm_allocator_read(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_2(0, 1), xm_dim_1(1), xm_dim_1(0),
	    buf1, buf2, 5);
	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_read(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	for (i = 0; i < 25; i++)
		if (!scalar_eq(-0.3*buf1[i], buf2[i]))
			fatal("%s: comparison failed", __func__);

	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_read(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_2(0, 1),
	    xm_dim_zero(0), buf1, buf2, 25);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_2(0, 1),
	    xm_dim_zero(0), buf2, buf1, 25);
	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_write(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	compare_tensors(t, u);

	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_read(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_2(1, 0),
	    xm_dim_zero(0), buf1, buf2, 25);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_2(1, 0),
	    xm_dim_zero(0), buf2, buf1, 25);
	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_write(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	compare_tensors(t, u);

	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_read(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_1(0),
	    xm_dim_1(1), buf1, buf2, 5);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_1(0),
	    xm_dim_1(1), buf2, buf1, 5);
	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_write(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	compare_tensors(t, u);

	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_read(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_1(1),
	    xm_dim_1(0), buf1, buf2, 5);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_1(1),
	    xm_dim_1(0), buf2, buf1, 5);
	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_write(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	compare_tensors(t, u);

	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_read(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_zero(0),
	    xm_dim_2(0, 1), buf1, buf2, 1);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_zero(0),
	    xm_dim_2(0, 1), buf2, buf1, 1);
	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_write(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	compare_tensors(t, u);

	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_read(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_2(0, 0), xm_dim_zero(0),
	    xm_dim_2(1, 0), buf1, buf2, 1);
	xm_tensor_fold_block(t, xm_dim_2(0, 0), xm_dim_zero(0),
	    xm_dim_2(1, 0), buf2, buf1, 1);
	ptr = xm_tensor_get_block_data_ptr(t, xm_dim_2(0, 0));
	xm_allocator_write(allocator_t, ptr, buf1, 25 * sizeof(xm_scalar_t));
	compare_tensors(t, u);

	free(buf1);
	free(buf2);
	xm_tensor_free_block_data(t);
	xm_tensor_free_block_data(u);
	xm_tensor_free(t);
	xm_tensor_free(u);
	xm_allocator_destroy(allocator_t);
	xm_allocator_destroy(allocator_u);
}

static void
unfold_test_3(const char *path)
{
	xm_allocator_t *allocator_t, *allocator_u;
	xm_block_space_t *bs;
	xm_tensor_t *t, *u;
	xm_scalar_t *buf1, *buf2;
	uintptr_t ptr;

	allocator_t = xm_allocator_create(path);
	assert(allocator_t);
	bs = xm_block_space_create(xm_dim_4(3, 4, 5, 6));
	assert(bs);
	t = xm_tensor_create(bs, allocator_t);
	assert(t);
	xm_block_space_free(bs);
	bs = NULL;
	ptr = xm_tensor_allocate_block_data(t, xm_dim_zero(4));
	xm_tensor_set_canonical_block(t, xm_dim_zero(4), ptr);
	fill_random(t);
	allocator_u = xm_allocator_create(NULL);
	assert(allocator_u);
	u = xm_tensor_clone(t, allocator_u);
	assert(u);
	buf1 = malloc(3*4*5*6*sizeof(xm_scalar_t));
	assert(buf1);
	buf2 = malloc(3*4*5*6*sizeof(xm_scalar_t));
	assert(buf2);

	xm_allocator_read(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_4(0, 1, 2, 3),
	    xm_dim_zero(0), buf1, buf2, 3*4*5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_4(0, 1, 2, 3),
	    xm_dim_zero(0), buf2, buf1, 3*4*5*6);
	xm_allocator_write(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	compare_tensors(t, u);

	xm_allocator_read(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_4(2, 0, 1, 3),
	    xm_dim_zero(0), buf1, buf2, 3*4*5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_4(2, 0, 1, 3),
	    xm_dim_zero(0), buf2, buf1, 3*4*5*6);
	xm_allocator_write(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	compare_tensors(t, u);

	xm_allocator_read(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_3(1, 3, 2),
	    xm_dim_1(0), buf1, buf2, 4*5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_3(1, 3, 2),
	    xm_dim_1(0), buf2, buf1, 4*5*6);
	xm_allocator_write(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	compare_tensors(t, u);

	xm_allocator_read(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_3(3, 2, 1),
	    xm_dim_1(0), buf1, buf2, 4*5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_3(3, 2, 1),
	    xm_dim_1(0), buf2, buf1, 4*5*6);
	xm_allocator_write(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	compare_tensors(t, u);

	xm_allocator_read(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_2(3, 2),
	    xm_dim_2(1, 0), buf1, buf2, 5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_2(3, 2),
	    xm_dim_2(1, 0), buf2, buf1, 5*6);
	xm_allocator_write(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	compare_tensors(t, u);

	xm_allocator_read(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_2(3, 2),
	    xm_dim_2(0, 1), buf1, buf2, 5*6);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_2(3, 2),
	    xm_dim_2(0, 1), buf2, buf1, 5*6);
	xm_allocator_write(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	compare_tensors(t, u);

	xm_allocator_read(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_1(2),
	    xm_dim_3(3, 0, 1), buf1, buf2, 5);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_1(2),
	    xm_dim_3(3, 0, 1), buf2, buf1, 5);
	xm_allocator_write(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	compare_tensors(t, u);

	xm_allocator_read(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_1(2),
	    xm_dim_3(1, 3, 0), buf1, buf2, 5);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_1(2),
	    xm_dim_3(1, 3, 0), buf2, buf1, 5);
	xm_allocator_write(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	compare_tensors(t, u);

	xm_allocator_read(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(0, 1, 2, 3), buf1, buf2, 1);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(0, 1, 2, 3), buf2, buf1, 1);
	xm_allocator_write(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	compare_tensors(t, u);

	xm_allocator_read(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(0, 3, 2, 1), buf1, buf2, 1);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(0, 3, 2, 1), buf2, buf1, 1);
	xm_allocator_write(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	compare_tensors(t, u);

	xm_allocator_read(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	xm_tensor_unfold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(1, 3, 2, 0), buf1, buf2, 1);
	xm_tensor_fold_block(t, xm_dim_zero(4), xm_dim_zero(0),
	    xm_dim_4(1, 3, 2, 0), buf2, buf1, 1);
	xm_allocator_write(allocator_t, ptr, buf1, 3*4*5*6*sizeof(xm_scalar_t));
	compare_tensors(t, u);

	free(buf1);
	free(buf2);
	xm_tensor_free_block_data(t);
	xm_tensor_free_block_data(u);
	xm_tensor_free(t);
	xm_tensor_free(u);
	xm_allocator_destroy(allocator_t);
	xm_allocator_destroy(allocator_u);
}

static unfold_test_fn unfold_tests[] = {
	unfold_test_1,
	unfold_test_2,
	unfold_test_3,
};

static void
test_dim(void)
{
	xm_dim_t idx1, idx2, dim;
	size_t i, offset;

	dim = xm_dim_4(5, 6, 3, 1);
	idx1 = xm_dim_zero(dim.n);
	while (xm_dim_ne(&idx1, &dim)) {
		offset = xm_dim_offset(&idx1, &dim);
		idx2 = xm_dim_from_offset(offset, &dim);
		if (xm_dim_ne(&idx1, &idx2))
			fatal("%s: dims do not match", __func__);
		xm_dim_inc(&idx1, &dim);
	}

	dim = xm_dim_3(1, 31, 16);
	idx1 = xm_dim_zero(dim.n);
	while (xm_dim_ne(&idx1, &dim)) {
		offset = xm_dim_offset(&idx1, &dim);
		idx2 = xm_dim_from_offset(offset, &dim);
		if (xm_dim_ne(&idx1, &idx2))
			fatal("%s: dims do not match", __func__);
		xm_dim_inc(&idx1, &dim);
	}

	i = 0;
	dim.n = 8;
	dim.i[0] = 5;
	dim.i[1] = 2;
	dim.i[2] = 3;
	dim.i[3] = 2;
	dim.i[4] = 1;
	dim.i[5] = 2;
	dim.i[6] = 5;
	dim.i[7] = 4;
	idx1 = xm_dim_zero(dim.n);
	while (xm_dim_ne(&idx1, &dim)) {
		offset = xm_dim_offset(&idx1, &dim);
		if (offset != i)
			fatal("%s: dims are not sequential", __func__);
		xm_dim_inc(&idx1, &dim);
		i++;
	}
}

int
main(void)
{
	const char *path = "xmpagefile";
	size_t i;

	printf("dim test 1... ");
	fflush(stdout);
	test_dim();
	printf("success\n");

	for (i = 0; i < sizeof unfold_tests / sizeof *unfold_tests; i++) {
		printf("unfold test %zu... ", i+1);
		fflush(stdout);
		unfold_tests[i](NULL);
		unfold_tests[i](path);
		printf("success\n");
	}
	for (i = 0; i < sizeof contract_tests / sizeof *contract_tests; i++) {
		printf("contract test %2zu... ", i+1);
		fflush(stdout);
		test_contract(&contract_tests[i], NULL, 0, 0);
		test_contract(&contract_tests[i], NULL, 0, random_scalar());
		test_contract(&contract_tests[i], NULL, random_scalar(), 0);
		test_contract(&contract_tests[i], NULL, random_scalar(),
		    random_scalar());
		test_contract(&contract_tests[i], path, 0, 0);
		test_contract(&contract_tests[i], path, 0, random_scalar());
		test_contract(&contract_tests[i], path, random_scalar(), 0);
		test_contract(&contract_tests[i], path, random_scalar(),
		    random_scalar());
		printf("success\n");
	}
	return 0;
}
