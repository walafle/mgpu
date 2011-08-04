// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BLAS_DETAIL_BLAS_ASYNC_IPP
#define MGPU_BLAS_DETAIL_BLAS_ASYNC_IPP

/**
 * @file blas_async.ipp
 *
 * This header contains the segmented blas asynchronous functions
 */

#include <mgpu/backend/blas.hpp>

namespace mgpu
{

namespace detail
{

// asynchronous free methods
// _____________________________________________________________________________


inline void alloc_single_blas_impl(backend::blas * b)
{
  b->allocate();
}

inline void free_single_blas_impl(backend::blas * b)
{
  // delete the handle
  if(b != NULL)
  {
    delete b;
  }
}


inline void blas_set_scalar_device_impl(backend::blas * b)
{
  b->set_scalar_device();
}

inline void blas_set_scalar_host_impl(backend::blas * b)
{
  b->set_scalar_host();
}

} // namespace detail

} // namespace mgpu



#endif // MGPU_BLAS_DETAIL_BLAS_ASYNC_IPP
