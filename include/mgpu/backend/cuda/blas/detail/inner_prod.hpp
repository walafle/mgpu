// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_BLAS_DETAIL_INNER_PROD_HPP
#define MGPU_BACKEND_CUDA_BLAS_DETAIL_INNER_PROD_HPP

/**
 * @file dot.hpp
 *
 * This header provides the implementations for the dot blas function
 */


#include <complex>

#include <cublas_v2.h>

#include <mgpu/core/dev_ptr.hpp>
#include <mgpu/container/iterator_traits.hpp>

namespace mgpu
{

namespace backend_detail
{

namespace cuda
{

namespace detail
{

template <typename T, typename ResultIterator>
void inner_prod(cublasHandle_t handle,
  const dev_ptr<T> & x, const dev_ptr<T> & y, ResultIterator & result,
  const std::size_t n,
  const std::size_t incx = 1, const std::size_t incy = 1)
{
  BOOST_ASSERT_MSG(false, "cuda blas inner_prod: unsupported type");
}

template <typename T, typename ResultIterator>
void inner_prod_c(cublasHandle_t handle,
  const dev_ptr<T> & x, const dev_ptr<T> & y, ResultIterator & result,
  const std::size_t n, const std::size_t incx = 1, const std::size_t incy = 1)
{
  BOOST_ASSERT_MSG(false, "cuda blas inner_prod_c: unsupported type");
}


} // namespace detail

} // namespace cuda

} // namespace backend_detail

} // namespace mgpu

#endif // MGPU_BACKEND_CUDA_BLAS_DETAIL_INNER_PROD_HPP
