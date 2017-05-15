/*
 * Copyright (c) 2014-2017 Ilya Kaliman <ilya.kaliman@gmail.com>
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "auxil.h"

xm_scalar_t
xm_random_scalar(void)
{
	double a = drand48() - 0.5;
#if defined(XM_SCALAR_DOUBLE_COMPLEX) || defined(XM_SCALAR_FLOAT_COMPLEX)
	double b = drand48() - 0.5;
	return ((xm_scalar_t)(a + b * I));
#else
	return ((xm_scalar_t)a);
#endif
}

static void
fatal(const char *msg)
{
	fprintf(stderr, "%s\n", msg);
	abort();
}

static uintptr_t
allocate_new_block(xm_allocator_t *allocator, const xm_dim_t *dim,
    int type)
{
	uintptr_t ptr;
	size_t size, size_bytes, i;
	xm_scalar_t *data;

	size = xm_dim_dot(dim);
	size_bytes = size * sizeof(xm_scalar_t);
	ptr = xm_allocate_block_data(allocator, dim);
	if (ptr == XM_NULL_PTR)
		fatal("cannot allocate tensor block data");
	if (type == XM_INIT_NONE)
		return (ptr);
	if (type == XM_INIT_RAND) {
		if ((data = malloc(size_bytes)) == NULL)
			fatal("cannot allocate memory");
		for (i = 0; i < size; i++)
			data[i] = xm_random_scalar();
		xm_allocator_write(allocator, ptr, data, size_bytes);
		free(data);
		return (ptr);
	}
	xm_allocator_memset(allocator, ptr, 0, size_bytes);
	return (ptr);
}

static xm_tensor_t *
create_tensor(xm_allocator_t *allocator, xm_dim_t nblocks, size_t blocksize)
{
	xm_tensor_t *tensor;
	xm_block_space_t *bs;
	xm_dim_t absdims;
	size_t i, j;

	absdims = xm_dim_scale(&nblocks, blocksize);
	if ((bs = xm_block_space_create(&absdims)) == NULL)
		fatal("cannot create a block-space");
	for (i = 0; i < nblocks.n; i++)
		for (j = 1; j < nblocks.i[i]; j++)
			xm_block_space_split(bs, i, j * blocksize);
	if ((tensor = xm_tensor_create(bs, "", allocator)) == NULL)
		fatal("cannot create a tensor");
	return tensor;
}

xm_tensor_t *
xm_aux_init(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx;
	size_t i, size;

	ret = create_tensor(allocator, nblocks, block_size);
	idx = xm_dim_zero(nblocks.n);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	size = xm_dim_dot(&nblocks);

	for (i = 0; i < size; i++) {
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);
		xm_dim_inc(&idx, &nblocks);
	}
	return (ret);
}

xm_tensor_t *
xm_aux_init_oo(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, perm2;

	assert(nblocks.n == 2);

	ret = create_tensor(allocator, nblocks, block_size);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	idx = xm_dim_zero(nblocks.n);
	perm = xm_dim_identity_permutation(nblocks.n);

	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
		if (xm_tensor_get_block_data_ptr(ret, &idx) != XM_NULL_PTR)
			continue;
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);
	} }
	return (ret);
}

xm_tensor_t *
xm_aux_init_ov(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx;

	assert(nblocks.n == 2);

	ret = create_tensor(allocator, nblocks, block_size);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	idx = xm_dim_zero(nblocks.n);

	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
		if (xm_tensor_get_block_data_ptr(ret, &idx) != XM_NULL_PTR)
			continue;
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);
	} }
	return (ret);
}

xm_tensor_t *
xm_aux_init_vv(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, perm2;

	assert(nblocks.n == 2);

	ret = create_tensor(allocator, nblocks, block_size);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	idx = xm_dim_zero(nblocks.n);
	perm = xm_dim_identity_permutation(nblocks.n);

	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
		if (xm_tensor_get_block_data_ptr(ret, &idx) != XM_NULL_PTR)
			continue;
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);
	} }
	return (ret);
}

xm_tensor_t *
xm_aux_init_vvx(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, perm2;

	assert(nblocks.n == 3);

	ret = create_tensor(allocator, nblocks, block_size);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	idx = xm_dim_zero(nblocks.n);
	perm = xm_dim_identity_permutation(nblocks.n);

	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < nblocks.i[2]; idx.i[2]++) {
		if (xm_tensor_get_block_data_ptr(ret, &idx) != XM_NULL_PTR)
			continue;
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);
	} } }
	return (ret);
}

xm_tensor_t *
xm_aux_init_oooo(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, perm2;

	assert(nblocks.n == 4);

	ret = create_tensor(allocator, nblocks, block_size);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	idx = xm_dim_zero(nblocks.n);
	perm = xm_dim_identity_permutation(nblocks.n);

	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < nblocks.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < nblocks.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(ret, &idx) != XM_NULL_PTR)
			continue;
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, 1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[2];
		idx2.i[1] = idx.i[3];
		idx2.i[2] = idx.i[0];
		idx2.i[3] = idx.i[1];
		perm2 = perm;
		perm2.i[0] = perm.i[2];
		perm2.i[1] = perm.i[3];
		perm2.i[2] = perm.i[0];
		perm2.i[3] = perm.i[1];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, 1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[3];
		idx2.i[1] = idx.i[2];
		idx2.i[2] = idx.i[0];
		idx2.i[3] = idx.i[1];
		perm2 = perm;
		perm2.i[0] = perm.i[3];
		perm2.i[1] = perm.i[2];
		perm2.i[2] = perm.i[0];
		perm2.i[3] = perm.i[1];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[2];
		idx2.i[1] = idx.i[3];
		idx2.i[2] = idx.i[1];
		idx2.i[3] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[2];
		perm2.i[1] = perm.i[3];
		perm2.i[2] = perm.i[1];
		perm2.i[3] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[3];
		idx2.i[1] = idx.i[2];
		idx2.i[2] = idx.i[1];
		idx2.i[3] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[3];
		perm2.i[1] = perm.i[2];
		perm2.i[2] = perm.i[1];
		perm2.i[3] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, 1.0);
	} } } }
	return (ret);
}

xm_tensor_t *
xm_aux_init_ooov(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, perm2;

	assert(nblocks.n == 4);

	ret = create_tensor(allocator, nblocks, block_size);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	idx = xm_dim_zero(nblocks.n);
	perm = xm_dim_identity_permutation(nblocks.n);

	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < nblocks.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < nblocks.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(ret, &idx) != XM_NULL_PTR)
			continue;
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);
	} } } }
	return (ret);
}

xm_tensor_t *
xm_aux_init_oovv(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, perm2;

	assert(nblocks.n == 4);

	ret = create_tensor(allocator, nblocks, block_size);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	idx = xm_dim_zero(nblocks.n);
	perm = xm_dim_identity_permutation(nblocks.n);

	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < nblocks.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < nblocks.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(ret, &idx) != XM_NULL_PTR)
			continue;
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, 1.0);
	} } } }
	return (ret);
}

xm_tensor_t *
xm_aux_init_ovov(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, perm2;

	assert(nblocks.n == 4);

	ret = create_tensor(allocator, nblocks, block_size);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	idx = xm_dim_zero(nblocks.n);
	perm = xm_dim_identity_permutation(nblocks.n);

	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < nblocks.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < nblocks.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(ret, &idx) != XM_NULL_PTR)
			continue;
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[2];
		idx2.i[1] = idx.i[3];
		idx2.i[2] = idx.i[0];
		idx2.i[3] = idx.i[1];
		perm2 = perm;
		perm2.i[0] = perm.i[2];
		perm2.i[1] = perm.i[3];
		perm2.i[2] = perm.i[0];
		perm2.i[3] = perm.i[1];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, 1.0);
	} } } }
	return (ret);
}

xm_tensor_t *
xm_aux_init_ovvv(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, perm2;

	assert(nblocks.n == 4);

	ret = create_tensor(allocator, nblocks, block_size);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	idx = xm_dim_zero(nblocks.n);
	perm = xm_dim_identity_permutation(nblocks.n);

	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < nblocks.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < nblocks.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(ret, &idx) != XM_NULL_PTR)
			continue;
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);
	} } } }
	return (ret);
}

xm_tensor_t *
xm_aux_init_vvvv(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, perm2;

	assert(nblocks.n == 4);

	ret = create_tensor(allocator, nblocks, block_size);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	idx = xm_dim_zero(nblocks.n);
	perm = xm_dim_identity_permutation(nblocks.n);

	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < nblocks.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < nblocks.i[3]; idx.i[3]++) {
		if (xm_tensor_get_block_data_ptr(ret, &idx) != XM_NULL_PTR)
			continue;
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		idx2.i[2] = idx.i[3];
		idx2.i[3] = idx.i[2];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		perm2.i[2] = perm.i[3];
		perm2.i[3] = perm.i[2];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, 1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[2];
		idx2.i[1] = idx.i[3];
		idx2.i[2] = idx.i[0];
		idx2.i[3] = idx.i[1];
		perm2 = perm;
		perm2.i[0] = perm.i[2];
		perm2.i[1] = perm.i[3];
		perm2.i[2] = perm.i[0];
		perm2.i[3] = perm.i[1];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, 1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[3];
		idx2.i[1] = idx.i[2];
		idx2.i[2] = idx.i[0];
		idx2.i[3] = idx.i[1];
		perm2 = perm;
		perm2.i[0] = perm.i[3];
		perm2.i[1] = perm.i[2];
		perm2.i[2] = perm.i[0];
		perm2.i[3] = perm.i[1];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[2];
		idx2.i[1] = idx.i[3];
		idx2.i[2] = idx.i[1];
		idx2.i[3] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[2];
		perm2.i[1] = perm.i[3];
		perm2.i[2] = perm.i[1];
		perm2.i[3] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[3];
		idx2.i[1] = idx.i[2];
		idx2.i[2] = idx.i[1];
		idx2.i[3] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[3];
		perm2.i[1] = perm.i[2];
		perm2.i[2] = perm.i[1];
		perm2.i[3] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, 1.0);
	} } } }
	return (ret);
}

xm_tensor_t *
xm_aux_init_ooovvv(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, perm2;

	assert(nblocks.n == 6);

	ret = create_tensor(allocator, nblocks, block_size);
	blk_dim = xm_dim_same(nblocks.n, block_size);
	idx = xm_dim_zero(nblocks.n);
	perm = xm_dim_identity_permutation(nblocks.n);

	for (idx.i[0] = 0; idx.i[0] < nblocks.i[0]; idx.i[0]++) {
	for (idx.i[1] = 0; idx.i[1] < nblocks.i[1]; idx.i[1]++) {
	for (idx.i[2] = 0; idx.i[2] < nblocks.i[2]; idx.i[2]++) {
	for (idx.i[3] = 0; idx.i[3] < nblocks.i[3]; idx.i[3]++) {
	for (idx.i[4] = 0; idx.i[4] < nblocks.i[4]; idx.i[4]++) {
	for (idx.i[5] = 0; idx.i[5] < nblocks.i[5]; idx.i[5]++) {
		if (xm_tensor_get_block_data_ptr(ret, &idx) != XM_NULL_PTR)
			continue;
		block = allocate_new_block(allocator, &blk_dim, type);
		xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[4] = idx.i[5];
		idx2.i[5] = idx.i[4];
		perm2 = perm;
		perm2.i[4] = perm.i[5];
		perm2.i[5] = perm.i[4];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, -1.0);

		idx2 = idx;
		idx2.i[0] = idx.i[1];
		idx2.i[1] = idx.i[0];
		idx2.i[4] = idx.i[5];
		idx2.i[5] = idx.i[4];
		perm2 = perm;
		perm2.i[0] = perm.i[1];
		perm2.i[1] = perm.i[0];
		perm2.i[4] = perm.i[5];
		perm2.i[5] = perm.i[4];
		if (xm_tensor_get_block_data_ptr(ret, &idx2) == XM_NULL_PTR)
			xm_tensor_set_block(ret, &idx2, &idx, &perm2, 1.0);
	} } } } } }
	return (ret);
}

xm_tensor_t *
xm_aux_init_13(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	xm_block_space_t *bs;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, absdims;

	(void)block_size;
	(void)nblocks;

	absdims = xm_dim_2(5, 5);
	if ((bs = xm_block_space_create(&absdims)) == NULL)
		fatal("xm_block_space_create");
	xm_block_space_split(bs, 0, 2);
	xm_block_space_split(bs, 1, 2);
	if ((ret = xm_tensor_create(bs, "", allocator)) == NULL)
		fatal("xm_tensor_create");

	blk_dim = xm_dim_2(2, 2);
	idx = xm_dim_2(0, 0);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	blk_dim = xm_dim_2(3, 3);
	idx = xm_dim_2(1, 1);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	blk_dim = xm_dim_2(2, 3);
	idx = xm_dim_2(0, 1);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	idx = xm_dim_2(1, 0);
	idx2 = xm_dim_2(0, 1);
	perm = xm_dim_2(1, 0);
	xm_tensor_set_block(ret, &idx, &idx2, &perm, 1.0);

	return (ret);
}

xm_tensor_t *
xm_aux_init_13c(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	xm_block_space_t *bs;
	uintptr_t block;
	xm_dim_t blk_dim, idx, absdims;

	(void)block_size;
	(void)nblocks;

	absdims = xm_dim_2(5, 5);
	if ((bs = xm_block_space_create(&absdims)) == NULL)
		fatal("xm_block_space_create");
	xm_block_space_split(bs, 0, 2);
	xm_block_space_split(bs, 1, 2);
	if ((ret = xm_tensor_create(bs, "", allocator)) == NULL)
		fatal("xm_tensor_create");

	blk_dim = xm_dim_2(2, 2);
	idx = xm_dim_2(0, 0);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	blk_dim = xm_dim_2(3, 3);
	idx = xm_dim_2(1, 1);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	blk_dim = xm_dim_2(2, 3);
	idx = xm_dim_2(0, 1);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	blk_dim = xm_dim_2(3, 2);
	idx = xm_dim_2(1, 0);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	return (ret);
}

xm_tensor_t *
xm_aux_init_14(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	xm_block_space_t *bs;
	uintptr_t block;
	xm_dim_t blk_dim, idx, idx2, perm, absdims;

	(void)block_size;
	(void)nblocks;

	absdims = xm_dim_4(6, 6, 6, 6);
	if ((bs = xm_block_space_create(&absdims)) == NULL)
		fatal("xm_block_space_create");
	xm_block_space_split(bs, 0, 2);
	xm_block_space_split(bs, 1, 2);
	xm_block_space_split(bs, 2, 2);
	xm_block_space_split(bs, 3, 2);
	if ((ret = xm_tensor_create(bs, "", allocator)) == NULL)
		fatal("xm_tensor_create");

	blk_dim = xm_dim_4(2, 2, 2, 2);
	idx = xm_dim_4(0, 0, 0, 0);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	blk_dim = xm_dim_4(4, 2, 2, 2);
	idx = xm_dim_4(1, 0, 0, 0);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	idx = xm_dim_4(0, 1, 0, 0);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(1, 0, 2, 3);
	xm_tensor_set_block(ret, &idx, &idx2, &perm, -1.0);

	blk_dim = xm_dim_4(4, 4, 2, 2);
	idx = xm_dim_4(1, 1, 0, 0);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	idx = xm_dim_4(0, 0, 1, 0);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_block(ret, &idx, &idx2, &perm, 1.0);

	blk_dim = xm_dim_4(4, 2, 4, 2);
	idx = xm_dim_4(1, 0, 1, 0);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	idx = xm_dim_4(0, 1, 1, 0);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(1, 0, 2, 3);
	xm_tensor_set_block(ret, &idx, &idx2, &perm, -1.0);

	blk_dim = xm_dim_4(4, 4, 4, 2);
	idx = xm_dim_4(1, 1, 1, 0);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	idx = xm_dim_4(0, 0, 0, 1);
	idx2 = xm_dim_4(1, 0, 0, 0);
	perm = xm_dim_4(2, 3, 1, 0);
	xm_tensor_set_block(ret, &idx, &idx2, &perm, -1.0);

	idx = xm_dim_4(1, 0, 0, 1);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(0, 1, 3, 2);
	xm_tensor_set_block(ret, &idx, &idx2, &perm, -1.0);

	idx = xm_dim_4(0, 1, 0, 1);
	idx2 = xm_dim_4(1, 0, 1, 0);
	perm = xm_dim_4(1, 0, 3, 2);
	xm_tensor_set_block(ret, &idx, &idx2, &perm, 1.0);

	idx = xm_dim_4(1, 1, 0, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(0, 1, 3, 2);
	xm_tensor_set_block(ret, &idx, &idx2, &perm, -1.0);

	idx = xm_dim_4(0, 0, 1, 1);
	idx2 = xm_dim_4(1, 1, 0, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_block(ret, &idx, &idx2, &perm, 1.0);

	idx = xm_dim_4(1, 0, 1, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(2, 3, 0, 1);
	xm_tensor_set_block(ret, &idx, &idx2, &perm, 1.0);

	idx = xm_dim_4(0, 1, 1, 1);
	idx2 = xm_dim_4(1, 1, 1, 0);
	perm = xm_dim_4(3, 2, 0, 1);
	xm_tensor_set_block(ret, &idx, &idx2, &perm, -1.0);

	blk_dim = xm_dim_4(4, 4, 4, 4);
	idx = xm_dim_4(1, 1, 1, 1);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	return (ret);
}

xm_tensor_t *
xm_aux_init_14b(xm_allocator_t *allocator, xm_dim_t nblocks,
    size_t block_size, int type)
{
	xm_tensor_t *ret;
	xm_block_space_t *bs;
	uintptr_t block;
	xm_dim_t blk_dim, idx, absdims;

	(void)block_size;
	(void)nblocks;

	absdims = xm_dim_4(3, 3, 6, 6);
	if ((bs = xm_block_space_create(&absdims)) == NULL)
		fatal("xm_block_space_create");
	xm_block_space_split(bs, 2, 2);
	xm_block_space_split(bs, 3, 2);
	if ((ret = xm_tensor_create(bs, "", allocator)) == NULL)
		fatal("xm_tensor_create");

	blk_dim = xm_dim_4(3, 3, 2, 2);
	idx = xm_dim_4(0, 0, 0, 0);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	blk_dim = xm_dim_4(3, 3, 2, 4);
	idx = xm_dim_4(0, 0, 0, 1);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	blk_dim = xm_dim_4(3, 3, 4, 2);
	idx = xm_dim_4(0, 0, 1, 0);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	blk_dim = xm_dim_4(3, 3, 4, 4);
	idx = xm_dim_4(0, 0, 1, 1);
	block = allocate_new_block(allocator, &blk_dim, type);
	xm_tensor_set_source_block(ret, &idx, &blk_dim, block);

	return (ret);
}
