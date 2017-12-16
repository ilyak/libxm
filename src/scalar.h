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

#ifndef XM_SCALAR_H_INCLUDED
#define XM_SCALAR_H_INCLUDED

#include <stddef.h>

#ifdef __cplusplus
#include <complex>
#else
#include <complex.h>
#endif

/** \file */

#ifdef __cplusplus
extern "C" {
#endif

/** Underlying type of the scalar. */
typedef enum {
	/** Scalar is float. */
	XM_SCALAR_FLOAT = 0,
	/** Scalar is float complex. */
	XM_SCALAR_FLOAT_COMPLEX,
	/** Scalar is double. */
	XM_SCALAR_DOUBLE,
	/** Scalar is double complex. */
	XM_SCALAR_DOUBLE_COMPLEX,
} xm_scalar_type_t;

/** A scalar value. It is defined as a largest floating point type convertible
 *  to all other types. */
#ifdef __cplusplus
typedef std::complex<double> xm_scalar_t;
#else
typedef double complex xm_scalar_t;
#endif

/** Return size of the scalar type in bytes.
 *  \param type The scalar type.
 *  \return Size of the scalar type in bytes. */
size_t xm_scalar_sizeof(xm_scalar_type_t type);

/** Set all elements of a vector to same scalar value.
 *  \param x Data vector.
 *  \param len Length of the vector in number of elements.
 *  \param type Scalar type of data.
 *  \param a Scalar value. */
void xm_scalar_set(void *x, size_t len, xm_scalar_type_t type, xm_scalar_t a);

/** Multiply all elements of a vector by a scalar.
 *  \param x Data vector.
 *  \param len Length of the vector in number of elements.
 *  \param type Scalar type of data.
 *  \param a Scalar factor. */
void xm_scalar_mul(void *x, size_t len, xm_scalar_type_t type, xm_scalar_t a);

/** Perform vector addition x = a * x + y.
 *  \param a Scalar value a.
 *  \param x Vector x.
 *  \param y Vector y.
 *  \param len Length of the vector in number of elements.
 *  \param type Scalar type of data. */
void xm_scalar_axpy(xm_scalar_t a, void *x, const void *y, size_t len,
    xm_scalar_type_t type);

/** Perform division of vector elements: x = x / (a * y).
 *  \param x Vector x.
 *  \param a Scalar value a.
 *  \param y Vector y.
 *  \param len Length of the vector in number of elements.
 *  \param type Scalar type of data. */
void xm_scalar_div(void *x, xm_scalar_t a, const void *y, size_t len,
    xm_scalar_type_t type);

/** Compute dot product of two vectors.
 *  \param x First vector.
 *  \param y Second vector.
 *  \param len Length of the vector in number of elements.
 *  \param type Scalar type of data.
 *  \return Dot product of the vectors. */
xm_scalar_t xm_scalar_dot(const void *x, const void *y, size_t len,
    xm_scalar_type_t type);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* XM_SCALAR_H_INCLUDED */
