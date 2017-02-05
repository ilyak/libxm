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
#include <err.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "alloc.h"

#define XM_MAX_BLKID (10ULL * 1024 * 1024)
#define XM_GROW_SIZE (256ULL * 1024 * 1024 * 1024)

struct xm_allocator {
	char *path;
	int fd, gen;
	size_t blkid, totalbytes, *blksize;
	off_t offset, *blkoffset;
};

static void *
xcalloc(size_t nmemb, size_t size)
{
	void *ptr;

	if ((ptr = calloc(nmemb, size)) == NULL)
		err(1, "xcalloc");
	return (ptr);
}

static void
extend_file(struct xm_allocator *allocator)
{
	allocator->totalbytes = allocator->totalbytes > XM_GROW_SIZE ?
	    allocator->totalbytes + XM_GROW_SIZE :
	    allocator->totalbytes * 2;
	if (ftruncate(allocator->fd, (off_t)allocator->totalbytes))
		err(1, "ftruncate");
}

static size_t
allocator_allocbytes(struct xm_allocator *allocator)
{
	size_t i, allocbytes = 0;

	for (i = 0; i < allocator->blkid; i++)
		allocbytes += allocator->blksize[i];
	return (allocbytes);
}

static void
copy_data(int dstfd, off_t dstoffset, int srcfd, off_t srcoffset, size_t size)
{
	ssize_t nread, nwrite;
	size_t totalbytes;
	off_t srcoff, dstoff;
	unsigned char buf[256*1024];

	totalbytes = 0;
	for (srcoff = srcoffset, dstoff = dstoffset;
	     totalbytes + sizeof buf < size;
	     srcoff += sizeof buf, dstoff += sizeof buf) {
		nread = pread(srcfd, buf, sizeof buf, srcoff);
		if (nread != (ssize_t)(sizeof buf))
			err(1, "pread");
		nwrite = pwrite(dstfd, buf, sizeof buf, dstoff);
		if (nwrite != (ssize_t)(sizeof buf))
			err(1, "pwrite");
		totalbytes += sizeof buf;
	}
	nread = pread(srcfd, buf, size - totalbytes, srcoff);
	if (nread != (ssize_t)(size - totalbytes))
		err(1, "pread");
	nwrite = pwrite(dstfd, buf, size - totalbytes, dstoff);
	if (nwrite != (ssize_t)(size - totalbytes))
		err(1, "pwrite");
}

static void
allocator_gc(struct xm_allocator *allocator)
{
	size_t i;
	off_t newoffset;
	int newfd;
	char buf[BUFSIZ];

	snprintf(buf, sizeof buf, "%s.%d", allocator->path, allocator->gen+1);
	if ((newfd = open(buf, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR)) == -1)
		err(1, "open");
	if (ftruncate(newfd, (off_t)allocator->totalbytes))
		err(1, "ftruncate");
	newoffset = 0;
	for (i = 0; i < allocator->blkid; i++) {
		if (allocator->blksize[i] == 0)
			continue;
		copy_data(newfd, newoffset, allocator->fd,
		    allocator->blkoffset[i], allocator->blksize[i]);
		allocator->blkoffset[i] = newoffset;
		newoffset += allocator->blksize[i];
	}
	if (close(allocator->fd))
		err(1, "close");
	snprintf(buf, sizeof buf, "%s.%d", allocator->path, allocator->gen);
	if (unlink(buf))
		err(1, "unlink");
	allocator->offset = newoffset;
	allocator->fd = newfd;
	allocator->gen++;
}

struct xm_allocator *
xm_allocator_create(const char *path)
{
	struct xm_allocator *allocator;
	char buf[BUFSIZ];

	allocator = xcalloc(1, sizeof *allocator);
	allocator->blksize = xcalloc(XM_MAX_BLKID,
	    sizeof *allocator->blksize);
	allocator->blkoffset = xcalloc(XM_MAX_BLKID,
	    sizeof *allocator->blkoffset);
	if (path == NULL)
		return (allocator);
	snprintf(buf, sizeof buf, "%s.%d", path, allocator->gen);
	if ((allocator->fd = open(buf, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR)) == -1)
		err(1, "open");
	allocator->totalbytes = 1024 * 1024; /* 1 MiB */
	if (ftruncate(allocator->fd, (off_t)allocator->totalbytes))
		err(1, "ftruncate");
	if ((allocator->path = strdup(path)) == NULL)
		err(1, "strdup");
	return (allocator);
}

const char *
xm_allocator_get_path(struct xm_allocator *allocator)
{
	return (allocator->path);
}

uintptr_t
xm_allocator_allocate(struct xm_allocator *allocator, size_t size_bytes)
{
	size_t blkid;

	if (allocator->path == NULL)
		return ((uintptr_t)(malloc(size_bytes)));
#pragma omp critical
{
	blkid = allocator->blkid;
	if (blkid == XM_MAX_BLKID)
		errx(1, "maximum number of blocks reached");
	if ((size_t)allocator->offset + size_bytes > allocator->totalbytes) {
		size_t allocbytes = allocator_allocbytes(allocator);
		if (allocbytes > allocator->totalbytes / 2)
			extend_file(allocator);
		else
			allocator_gc(allocator);
	}
	allocator->blkoffset[blkid] = allocator->offset;
	allocator->offset += size_bytes;
	allocator->blksize[blkid] = size_bytes;
	allocator->blkid++;
}
	return (blkid);
}

void
xm_allocator_memset(struct xm_allocator *allocator, uintptr_t data_ptr,
    unsigned char c, size_t size_bytes)
{
	size_t blkid, totalbytes;
	ssize_t nwrite;
	off_t offset;
	unsigned char buf[256*1024];

	assert(data_ptr != XM_NULL_PTR);

	if (allocator->path == NULL) {
		memset((void *)data_ptr, c, size_bytes);
		return;
	}
	blkid = (size_t)data_ptr;
	memset(buf, c, sizeof buf);
	totalbytes = 0;
	for (offset = allocator->blkoffset[blkid];
	     totalbytes + sizeof buf < size_bytes;
	     offset += sizeof buf) {
		nwrite = pwrite(allocator->fd, buf, sizeof buf, offset);
		if (nwrite != (ssize_t)(sizeof buf))
			err(1, "pwrite");
		totalbytes += sizeof buf;
	}
	nwrite = pwrite(allocator->fd, buf, size_bytes - totalbytes, offset);
	if (nwrite != (ssize_t)(size_bytes - totalbytes))
		err(1, "pwrite");
}

void
xm_allocator_read(struct xm_allocator *allocator, uintptr_t data_ptr,
    void *mem, size_t size_bytes)
{
	size_t blkid;
	ssize_t nread;

	assert(data_ptr != XM_NULL_PTR);

	if (allocator->path == NULL) {
		memcpy(mem, (const void *)data_ptr, size_bytes);
		return;
	}
	blkid = (size_t)data_ptr;
	nread = pread(allocator->fd, mem, size_bytes,
	    allocator->blkoffset[blkid]);
	if (nread != (ssize_t)size_bytes)
		err(1, "pread");
}

void
xm_allocator_write(struct xm_allocator *allocator, uintptr_t data_ptr,
    const void *mem, size_t size_bytes)
{
	size_t blkid;
	ssize_t nwrite;

	assert(data_ptr != XM_NULL_PTR);

	if (allocator->path == NULL) {
		memcpy((void *)data_ptr, mem, size_bytes);
		return;
	}
	blkid = (size_t)data_ptr;
	nwrite = pwrite(allocator->fd, mem, size_bytes,
	    allocator->blkoffset[blkid]);
	if (nwrite != (ssize_t)size_bytes)
		err(1, "pwrite");
}

void
xm_allocator_deallocate(struct xm_allocator *allocator, uintptr_t data_ptr)
{
	size_t blkid;

	if (data_ptr == XM_NULL_PTR)
		return;
	if (allocator->path == NULL) {
		free((void *)data_ptr);
		return;
	}
	blkid = (size_t)data_ptr;
	allocator->blksize[blkid] = 0;
	allocator->blkoffset[blkid] = (off_t)-1;
}

void
xm_allocator_destroy(struct xm_allocator *allocator)
{
	char buf[BUFSIZ];

	if (allocator) {
		if (allocator->path) {
			snprintf(buf, sizeof buf, "%s.%d", allocator->path,
			    allocator->gen);
			if (close(allocator->fd))
				err(1, "close");
			if (unlink(buf))
				err(1, "unlink");
			free(allocator->path);
		}
		free(allocator->blksize);
		free(allocator->blkoffset);
		free(allocator);
	}
}
