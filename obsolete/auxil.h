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

#ifndef XM_AUXIL_H
#define XM_AUXIL_H

#include "xm.h"

#define XM_INIT_NONE    0  /* Do not initialize memory. */
#define XM_INIT_ZERO    1  /* Initialize with zeros. */
#define XM_INIT_RAND    2  /* Initialize with random data. */

xm_tensor_t *xm_aux_init(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_oo(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_ov(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_vv(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_vvx(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_oooo(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_ooov(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_oovv(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_ovov(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_ovvv(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_vvvv(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_ooovvv(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_13(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_13c(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_14(xm_allocator_t *, xm_dim_t, size_t, int);
xm_tensor_t *xm_aux_init_14b(xm_allocator_t *, xm_dim_t, size_t, int);
xm_scalar_t xm_random_scalar(void);

#endif /* XM_AUXIL_H */
