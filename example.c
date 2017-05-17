#include "xm.h"

#include <stdio.h>

static void
print_tensor(const xm_tensor_t *t)
{
	xm_dim_t absdims = xm_tensor_get_abs_dims(t);
	xm_dim_t idx = xm_dim_zero(2);
	for (idx.i[0] = 0; idx.i[0] < absdims.i[0]; idx.i[0]++) {
		for (idx.i[1] = 0; idx.i[1] < absdims.i[1]; idx.i[1]++) {
			xm_scalar_t el;
			el = xm_tensor_get_abs_element(t, &idx);
			printf(" % 6.2lf", el);
		}
		printf("\n");
	}
}

int
main(void)
{
	/* Create an allocator. Passing NULL means that we store data in RAM,
	 * not in a file on disk. */
	xm_allocator_t *allocator = xm_allocator_create(NULL);

	/* Our matrices will be 4x5, 5x3, and 4x3 elements in size. */
	xm_dim_t absdimsa = xm_dim_2(4, 5);
	xm_dim_t absdimsb = xm_dim_2(5, 3);
	xm_dim_t absdimsc = xm_dim_2(4, 3);

	/* Create the block-spaces. */
	xm_block_space_t *bsa = xm_block_space_create(&absdimsa);
	xm_block_space_t *bsb = xm_block_space_create(&absdimsb);
	xm_block_space_t *bsc = xm_block_space_create(&absdimsc);

	/* Split the block-spaces into blocks. Block-spaces must be consistent
	 * with each other. */
	/* block-space a - 6 blocks */
	xm_block_space_split(bsa, 0, 2);
	xm_block_space_split(bsa, 1, 2);
	xm_block_space_split(bsa, 1, 3);
	/* block-space b - 3 blocks */
	xm_block_space_split(bsb, 0, 2);
	xm_block_space_split(bsb, 0, 3);
	/* block-space c - 2 blocks */
	xm_block_space_split(bsc, 0, 2);

	/* Create tensors a, b, c. Tensors are initialized with all
	 * zero-blocks by default. */
	xm_tensor_t *a = xm_tensor_create(bsa, allocator);
	xm_tensor_t *b = xm_tensor_create(bsb, allocator);
	xm_tensor_t *c = xm_tensor_create(bsc, allocator);

	/* Fill a and b with some data. */
	xm_dim_t ii, jj, perm;
	uintptr_t data_ptr;

	/* tensor a */
	xm_scalar_t blka[] = { 1, 2, 3, 4 };
	ii = xm_dim_2(0, 0);
	data_ptr = xm_tensor_allocate_block_data(a, &ii);
	xm_allocator_write(allocator, data_ptr, blka, sizeof blka);
	xm_tensor_set_source_block(a, &ii, data_ptr);
	/* second block is transposed and negated first one */
	jj = xm_dim_2(1, 2);
	perm = xm_dim_2(1, 0);
	xm_tensor_set_block(a, &jj, &ii, &perm, -1.0);

	/* tensor b */
	xm_scalar_t blkb[] = { 6, 5, -4, 3, 2, -1 };
	ii = xm_dim_2(0, 0);
	data_ptr = xm_tensor_allocate_block_data(b, &ii);
	xm_allocator_write(allocator, data_ptr, blkb, sizeof blkb);
	xm_tensor_set_source_block(b, &ii, data_ptr);
	/* second block is a copy of the first one multiplied by -0.5 */
	jj = xm_dim_2(2, 0);
	perm = xm_dim_identity_permutation(perm.n);
	xm_tensor_set_block(b, &jj, &ii, &perm, -0.5);

	/* The result c must be allocated explicitly. */
	ii = xm_dim_2(0, 0);
	data_ptr = xm_tensor_allocate_block_data(c, &ii);
	xm_tensor_set_source_block(c, &ii, data_ptr);
	ii = xm_dim_2(1, 0);
	data_ptr = xm_tensor_allocate_block_data(c, &ii);
	xm_tensor_set_source_block(c, &ii, data_ptr);

	/* Compute c = 2*a*b */
	xm_contract(2.0, a, b, 0.0, c, "ik", "kj", "ij");

	/* Print the result. */
	printf("tensor a\n");
	print_tensor(a);
	printf("\ntensor b\n");
	print_tensor(b);
	printf("\ntensor c = 2*a*b\n");
	print_tensor(c);

	/* Finally, cleanup all allocated resources. */
	xm_tensor_free_block_data(a);
	xm_tensor_free_block_data(b);
	xm_tensor_free_block_data(c);
	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_tensor_free(c);
	xm_block_space_free(bsa);
	xm_block_space_free(bsb);
	xm_block_space_free(bsc);
	xm_allocator_destroy(allocator);
	return 0;
}
