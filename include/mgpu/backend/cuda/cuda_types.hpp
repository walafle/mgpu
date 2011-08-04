// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_CUDA_TYPES_HPP
#define MGPU_BACKEND_CUDA_CUDA_TYPES_HPP

/**
 * @file cuda_types.hpp
 *
 * This header provides means of translating types to CUDA types if necessary
 */

#include <cuComplex.h>
#include <mgpu/backend/cuda/complex.hpp>

namespace mgpu
{

namespace backend_detail
{

namespace cuda
{


// map regular types to cuda blas types -----

template <typename T>
struct cuda_type
{
  typedef T type;
};

template<>
struct cuda_type<std::complex<float> >
{
  typedef cuComplex type;
};

template<>
struct cuda_type<std::complex<double> >
{
  typedef cuDoubleComplex type;
};


} // namespace cuda

} // namespace backend_detail

} // namespace mgpu

#endif // MGPU_BACKEND_CUDA_CUDA_TYPES_HPP
