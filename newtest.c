#include "xm.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*make_abc_fn)(xm_allocator_t *, xm_tensor_t **, xm_tensor_t **,
    xm_tensor_t **);

struct test {
	make_abc_fn make_abc;
	const char *idxa, *idxb, *idxc;
};

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
	xm_dim_t blkdims, idx, nblocks;
	xm_scalar_t *data;
	size_t i, blksize, maxblksize;
	uintptr_t ptr;

	allocator = xm_tensor_get_allocator(t);
	bs = xm_tensor_get_block_space(t);
	maxblksize = xm_block_space_get_largest_block_size(bs);
	data = malloc(maxblksize * sizeof(xm_scalar_t));
	assert(data);
	nblocks = xm_block_space_get_nblocks(bs);
	for (idx = xm_dim_zero(nblocks.n);
	     xm_dim_ne(&idx, &nblocks);
	     xm_dim_inc(&idx, &nblocks)) {
		ptr = xm_tensor_get_block_data_ptr(t, &idx);
		if (ptr == XM_NULL_PTR)
			continue;
		blkdims = xm_tensor_get_block_dims(t, &idx);
		blksize = xm_dim_dot(&blkdims);
		for (i = 0; i < blksize; i++)
			data[i] = random_scalar();
		xm_allocator_write(allocator, ptr, data,
		    blksize * sizeof(xm_scalar_t));
	}
	free(data);
}

static void
parse_idx(const char *str1, const char *str2, xm_dim_t *mask1, xm_dim_t *mask2)
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

	parse_idx(idxa, idxb, &cidxa, &cidxb);
	parse_idx(idxc, idxa, &cidxc, &aidxa);
	parse_idx(idxc, idxb, &aidxc, &aidxb);
	absdimsa = xm_tensor_get_abs_dims(a);
	absdimsb = xm_tensor_get_abs_dims(b);
	absdimsc = xm_tensor_get_abs_dims(c);
	nk = xm_dim_dot_mask(&absdimsa, &cidxa);
	for (ic = xm_dim_zero(absdimsc.n);
	     xm_dim_ne(&ic, &absdimsc);
	     xm_dim_inc(&ic, &absdimsc)) {
		ref = beta * xm_tensor_get_element(c, &ic);
		ia = xm_dim_zero(absdimsa.n);
		ib = xm_dim_zero(absdimsb.n);
		xm_dim_set_mask(&ia, &aidxa, &ic, &cidxc);
		xm_dim_set_mask(&ib, &aidxb, &ic, &aidxc);
		for (k = 0; k < nk; k++) {
			xm_scalar_t ea = xm_tensor_get_element(a, &ia);
			xm_scalar_t eb = xm_tensor_get_element(b, &ib);
			ref += alpha * ea * eb;
			xm_dim_inc_mask(&ia, &absdimsa, &cidxa);
			xm_dim_inc_mask(&ib, &absdimsb, &cidxb);
		}
		ecc = xm_tensor_get_element(cc, &ic);
		if (!scalar_eq(ecc, ref)) {
			printf("result != ref\n");
			abort();
		}
	}
}

static void
test_contract(struct test *test, const char *path, xm_scalar_t alpha,
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
	ptr = xm_tensor_allocate_block_data(a, &idx);
	xm_tensor_set_canonical_block(a, &idx, ptr);
	ptr = xm_tensor_allocate_block_data(b, &idx);
	xm_tensor_set_canonical_block(b, &idx, ptr);
	ptr = xm_tensor_allocate_block_data(c, &idx);
	xm_tensor_set_canonical_block(c, &idx, ptr);
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
		ptr = xm_tensor_allocate_block_data(a, &idx);
		xm_tensor_set_canonical_block(a, &idx, ptr);
		ptr = xm_tensor_allocate_block_data(b, &idx);
		xm_tensor_set_canonical_block(b, &idx, ptr);
		ptr = xm_tensor_allocate_block_data(c, &idx);
		xm_tensor_set_canonical_block(c, &idx, ptr);
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
	ptr = xm_tensor_allocate_block_data(a, &idx);
	xm_tensor_set_canonical_block(a, &idx, ptr);
	ptr = xm_tensor_allocate_block_data(b, &idx);
	xm_tensor_set_canonical_block(b, &idx, ptr);
	idx = xm_dim_zero(2);
	ptr = xm_tensor_allocate_block_data(c, &idx);
	xm_tensor_set_canonical_block(c, &idx, ptr);
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
	ptr = xm_tensor_allocate_block_data(a, &idx);
	xm_tensor_set_canonical_block(a, &idx, ptr);
	ptr = xm_tensor_allocate_block_data(b, &idx);
	xm_tensor_set_canonical_block(b, &idx, ptr);
	ptr = xm_tensor_allocate_block_data(c, &idx);
	xm_tensor_set_canonical_block(c, &idx, ptr);
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
	ptr = xm_tensor_allocate_block_data(a, &idx);
	xm_tensor_set_canonical_block(a, &idx, ptr);
	ptr = xm_tensor_allocate_block_data(b, &idx);
	xm_tensor_set_canonical_block(b, &idx, ptr);
	ptr = xm_tensor_allocate_block_data(c, &idx);
	xm_tensor_set_canonical_block(c, &idx, ptr);
	idx = xm_dim_2(0, 1);
	ptr = xm_tensor_allocate_block_data(c, &idx);
	xm_tensor_set_canonical_block(c, &idx, ptr);
	idx = xm_dim_2(1, 0);
	ptr = xm_tensor_allocate_block_data(c, &idx);
	xm_tensor_set_canonical_block(c, &idx, ptr);
	idx = xm_dim_2(1, 1);
	ptr = xm_tensor_allocate_block_data(a, &idx);
	xm_tensor_set_canonical_block(a, &idx, ptr);
	ptr = xm_tensor_allocate_block_data(b, &idx);
	xm_tensor_set_canonical_block(b, &idx, ptr);
	ptr = xm_tensor_allocate_block_data(c, &idx);
	xm_tensor_set_canonical_block(c, &idx, ptr);
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
	ptr = xm_tensor_allocate_block_data(a, &idx);
	xm_tensor_set_canonical_block(a, &idx, ptr);
	idx = xm_dim_1(1);
	ptr = xm_tensor_allocate_block_data(a, &idx);
	xm_tensor_set_canonical_block(a, &idx, ptr);

	idx = xm_dim_2(0, 0);
	ptr = xm_tensor_allocate_block_data(b, &idx);
	xm_tensor_set_canonical_block(b, &idx, ptr);
	idx = xm_dim_2(0, 1);
	ptr = xm_tensor_allocate_block_data(b, &idx);
	xm_tensor_set_canonical_block(b, &idx, ptr);

	idx = xm_dim_3(0, 0, 0);
	ptr = xm_tensor_allocate_block_data(c, &idx);
	xm_tensor_set_canonical_block(c, &idx, ptr);
	idx = xm_dim_3(0, 0, 1);
	ptr = xm_tensor_allocate_block_data(c, &idx);
	xm_tensor_set_canonical_block(c, &idx, ptr);
	idx = xm_dim_3(0, 1, 0);
	ptr = xm_tensor_allocate_block_data(c, &idx);
	xm_tensor_set_canonical_block(c, &idx, ptr);
	idx = xm_dim_3(0, 1, 1);
	ptr = xm_tensor_allocate_block_data(c, &idx);
	xm_tensor_set_canonical_block(c, &idx, ptr);
	*aa = a;
	*bb = b;
	*cc = c;
}

static void
make_abc_7(xm_allocator_t *allocator, xm_tensor_t **aa, xm_tensor_t **bb,
    xm_tensor_t **cc)
{
	xm_dim_t idx, canidx, perm, nblocks;
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
	ptr = xm_tensor_allocate_block_data(a, &idx);
	xm_tensor_set_canonical_block(a, &idx, ptr);
	idx = xm_dim_3(1, 0, 0);
	ptr = xm_tensor_allocate_block_data(a, &idx);
	xm_tensor_set_canonical_block(a, &idx, ptr);
	idx = xm_dim_3(2, 0, 0);
	canidx = xm_dim_3(0, 0, 0);
	perm = xm_dim_3(2, 1, 0);
	xm_tensor_set_derivative_block(a, &idx, &canidx, &perm, -0.5);

	bsb = xm_block_space_create(xm_dim_3(7, 4, 8));
	xm_block_space_split(bsb, 0, 2);
	xm_block_space_split(bsb, 0, 5);
	xm_block_space_split(bsb, 1, 2);
	xm_block_space_split(bsb, 2, 2);
	xm_block_space_split(bsb, 2, 6);
	b = xm_tensor_create(bsb, allocator);
	xm_block_space_free(bsb);

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
		ptr = xm_tensor_allocate_block_data(c, &idx);
		xm_tensor_set_canonical_block(c, &idx, ptr);
	}
	*aa = a;
	*bb = b;
	*cc = c;
}

static struct test tests[] = {
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
};

int
main(void)
{
	const char *path = "xmpagefile";
	size_t i;

	for (i = 0; i < sizeof tests / sizeof *tests; i++) {
		printf("Test contract %2zu... ", i+1);
		fflush(stdout);
		test_contract(&tests[i], NULL, 0, 0);
		test_contract(&tests[i], NULL, 0, random_scalar());
		test_contract(&tests[i], NULL, random_scalar(), 0);
		test_contract(&tests[i], NULL, random_scalar(),
		    random_scalar());
		test_contract(&tests[i], path, 0, 0);
		test_contract(&tests[i], path, 0, random_scalar());
		test_contract(&tests[i], path, random_scalar(), 0);
		test_contract(&tests[i], path, random_scalar(),
		    random_scalar());
		printf("success\n");
	}
	return 0;
}
