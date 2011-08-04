// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_COMPLEX_HPP
#define MGPU_BACKEND_CUDA_COMPLEX_HPP

/**
 * @file complex.hpp
 *
 * This header provides operators for cuda complex types
 */

#include <cuComplex.h>


namespace mgpu
{

namespace backend_detail
{

namespace cuda
{

// complex float -----

/// multiply two complex floats
inline __device__ cuFloatComplex operator* (cuFloatComplex a, cuFloatComplex b)
{ return make_cuFloatComplex(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }

/// add two complex floats
inline __device__ cuFloatComplex operator+ (cuFloatComplex a, cuFloatComplex b)
{ return make_cuFloatComplex(a.x + b.x, a.y + b.y); }

/// addition assign a complex float to another
inline __device__ cuFloatComplex operator+= (cuFloatComplex a, cuFloatComplex b)
{ return make_cuFloatComplex(a.x + b.x, a.y + b.y); }

/// multiply a complex float with a real float
inline __device__ cuFloatComplex operator*( cuFloatComplex a, float  b)
{ return make_cuFloatComplex( b*a.x, b*a.y ); }

/// subtract a complex float from another
inline __device__ cuFloatComplex operator-( cuFloatComplex a, cuFloatComplex b)
{ return make_cuFloatComplex( a.x - b.x, a.y - b.y ); }



// complex double -----

/// multiply two complex floats
inline __device__ cuDoubleComplex operator* (cuDoubleComplex a,
  cuDoubleComplex b)
{ return make_cuDoubleComplex(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }

/// add two complex floats
inline __device__ cuDoubleComplex operator+ (cuDoubleComplex a,
  cuDoubleComplex b)
{ return make_cuDoubleComplex(a.x + b.x, a.y + b.y); }

/// addition assign a complex float to another
inline __device__ cuDoubleComplex operator+= (cuDoubleComplex a,
  cuDoubleComplex b)
{ return make_cuDoubleComplex(a.x + b.x, a.y + b.y); }

/// multiply a complex float with a real float
inline __device__ cuDoubleComplex operator*( cuDoubleComplex a, float  b)
{ return make_cuDoubleComplex( b*a.x, b*a.y ); }

/// subtract a complex float from another
inline __device__ cuDoubleComplex operator-( cuDoubleComplex a,
  cuDoubleComplex b)
{ return make_cuDoubleComplex( a.x - b.x, a.y - b.y ); }


} // namespace cuda

} // namespace backend_detail

} // namespace mgpu

#endif // MGPU_BACKEND_CUDA_COMPLEX_HPP
