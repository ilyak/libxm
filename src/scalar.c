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

#include <string.h>

#include "scalar.h"
#include "util.h"

size_t
xm_scalar_sizeof(xm_scalar_type_t type)
{
	static const size_t tbl[] = {
		sizeof(float),
		sizeof(float complex),
		sizeof(double),
		sizeof(double complex),
	};
	return tbl[type];
}

void
xm_scalar_set(void *x, xm_scalar_t a, size_t len, xm_scalar_type_t type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xx = x;
		for (i = 0; i < len; i++)
			xx[i] = a;
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xx = x;
		for (i = 0; i < len; i++)
			xx[i] = a;
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xx = x;
		for (i = 0; i < len; i++)
			xx[i] = a;
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xx = x;
		for (i = 0; i < len; i++)
			xx[i] = a;
		return;
	}
	}
}

void
xm_scalar_scale(void *x, xm_scalar_t a, size_t len, xm_scalar_type_t type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xx = x;
		for (i = 0; i < len; i++)
			xx[i] *= a;
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xx = x;
		for (i = 0; i < len; i++)
			xx[i] *= a;
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xx = x;
		for (i = 0; i < len; i++)
			xx[i] *= a;
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xx = x;
		for (i = 0; i < len; i++)
			xx[i] *= a;
		return;
	}
	}
}

void
xm_scalar_axpy(void *x, xm_scalar_t a, const void *y, xm_scalar_t b, size_t len,
    xm_scalar_type_t type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xx = x;
		const float *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = a * xx[i] + b * yy[i];
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xx = x;
		const float complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = a * xx[i] + b * yy[i];
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xx = x;
		const double *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = a * xx[i] + b * yy[i];
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xx = x;
		const double complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = a * xx[i] + b * yy[i];
		return;
	}
	}
}

void
xm_scalar_mul(void *x, xm_scalar_t a, const void *y, size_t len,
    xm_scalar_type_t type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xx = x;
		const float *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] * (a * yy[i]);
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xx = x;
		const float complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] * (a * yy[i]);
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xx = x;
		const double *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] * (a * yy[i]);
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xx = x;
		const double complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] * (a * yy[i]);
		return;
	}
	}
}

void
xm_scalar_div(void *x, xm_scalar_t a, const void *y, size_t len,
    xm_scalar_type_t type)
{
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		float *xx = x;
		const float *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] / (a * yy[i]);
		return;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		float complex *xx = x;
		const float complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] / (a * yy[i]);
		return;
	}
	case XM_SCALAR_DOUBLE: {
		double *xx = x;
		const double *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] / (a * yy[i]);
		return;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		double complex *xx = x;
		const double complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = xx[i] / (a * yy[i]);
		return;
	}
	}
}

xm_scalar_t
xm_scalar_dot(const void *x, const void *y, size_t len, xm_scalar_type_t type)
{
	xm_scalar_t dot = 0;
	size_t i;

	switch (type) {
	case XM_SCALAR_FLOAT: {
		const float *xx = x;
		const float *yy = y;
		for (i = 0; i < len; i++)
			dot += xx[i] * yy[i];
		return dot;
	}
	case XM_SCALAR_FLOAT_COMPLEX: {
		const float complex *xx = x;
		const float complex *yy = y;
		for (i = 0; i < len; i++)
			dot += xx[i] * yy[i];
		return dot;
	}
	case XM_SCALAR_DOUBLE: {
		const double *xx = x;
		const double *yy = y;
		for (i = 0; i < len; i++)
			dot += xx[i] * yy[i];
		return dot;
	}
	case XM_SCALAR_DOUBLE_COMPLEX: {
		const double complex *xx = x;
		const double complex *yy = y;
		for (i = 0; i < len; i++)
			dot += xx[i] * yy[i];
		return dot;
	}
	}
	return dot;
}

void
xm_scalar_convert(void *x, const void *y, size_t len, xm_scalar_type_t xtype,
    xm_scalar_t ytype)
{
	size_t i;

	if (xtype == ytype) {
		memcpy(x, y, len * xm_scalar_sizeof(xtype));
		return;
	}
	if (xtype == XM_SCALAR_DOUBLE && ytype == XM_SCALAR_FLOAT) {
		double *xx = x;
		const float *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = yy[i];
		return;
	}
	if (xtype == XM_SCALAR_FLOAT && ytype == XM_SCALAR_DOUBLE) {
		float *xx = x;
		const double *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = yy[i];
		return;
	}
	if (xtype == XM_SCALAR_DOUBLE_COMPLEX &&
	    ytype == XM_SCALAR_FLOAT_COMPLEX) {
		double complex *xx = x;
		const float complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = yy[i];
		return;
	}
	if (xtype == XM_SCALAR_FLOAT_COMPLEX &&
	    ytype == XM_SCALAR_DOUBLE_COMPLEX) {
		float complex *xx = x;
		const double complex *yy = y;
		for (i = 0; i < len; i++)
			xx[i] = yy[i];
		return;
	}
	fatal("unsupported scalar conversion");
}
