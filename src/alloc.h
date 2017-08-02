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

#ifndef XM_ALLOC_H
#define XM_ALLOC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XM_NULL_PTR ((uint64_t)(-1))

/* MPI-aware thread-safe disk-backed memory allocator. */
typedef struct xm_allocator xm_allocator_t;

xm_allocator_t *xm_allocator_create(const char *path);
const char *xm_allocator_get_path(xm_allocator_t *allocator);
uint64_t xm_allocator_allocate(xm_allocator_t *allocator,
    size_t size_bytes);
void xm_allocator_read(xm_allocator_t *allocator, uint64_t data_ptr,
    void *mem, size_t size_bytes);
void xm_allocator_write(xm_allocator_t *allocator, uint64_t data_ptr,
    const void *mem, size_t size_bytes);
void xm_allocator_deallocate(xm_allocator_t *allocator, uint64_t data_ptr);
void xm_allocator_destroy(xm_allocator_t *allocator);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_ALLOC_H */
