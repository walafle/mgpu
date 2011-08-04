// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_BLAS_BLAS_HPP
#define MGPU_BACKEND_CUDA_BLAS_BLAS_HPP

/**
 * @file blas.hpp
 *
 * This header provides the mighty blas class
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <mgpu/core/dev_ptr.hpp>
#include <mgpu/container/range_traits.hpp>
#include <mgpu/container/iterator_traits.hpp>
#include <mgpu/backend/cuda/cuda_call.hpp>
#include <mgpu/backend/cuda/dev_stream.hpp>
#include <mgpu/backend/cuda/blas/exception.hpp>

#include <mgpu/backend/cuda/blas/blas_traits.hpp>
#include <mgpu/backend/cuda/blas/detail/axpy.hpp>

namespace mgpu
{

namespace backend_detail
{

namespace cuda
{

/**
 * @brief the blas class wraps all blas functionality
 */
class blas
{
  public:

    /// create a blas object
    inline blas() { allocate(); }

    /// create a blas object but defer allocation of blas handle
    explicit inline blas(bool deferred_allocation)
    {
      if(!deferred_allocation)
      {
        allocate();
      }
    }

    /// destroy the blas object
    inline ~blas() { MGPU_CUDA_BLAS_CALL(cublasDestroy(handle_)); }

    /// allocate the blas object
    inline void allocate()
    { MGPU_CUDA_BLAS_CALL(cublasCreate(&handle_)); }

    /// associate all following blas commands with stream
    inline void set_stream(dev_stream & stream)
    { MGPU_CUDA_BLAS_CALL(cublasSetStream(handle_, stream.get())); }

    /// associate all following blas commands with default stream
    inline void reset_stream()
    { MGPU_CUDA_BLAS_CALL(cublasSetStream(handle_, NULL)); }

    /// synchronize all operations on the current stream
    inline void sync()
    {
      cudaStream_t stream;
      MGPU_CUDA_BLAS_CALL(cublasGetStream(handle_, &stream));
      MGPU_CUDA_CALL(cudaStreamSynchronize(stream));
    }

    /// indicate that scalar values are passed as reference on host
    inline void set_scalar_device()
    {
      MGPU_CUDA_BLAS_CALL(cublasSetPointerMode(handle_,
        CUBLAS_POINTER_MODE_DEVICE));
    }

    /// indicate that scalar values are passed as reference on device
    inline void set_scalar_host()
    {
      MGPU_CUDA_BLAS_CALL(cublasSetPointerMode(handle_,
        CUBLAS_POINTER_MODE_HOST));
    }

    /// calculate inner product product
    template <typename XRange, typename YRange, typename ResultIterator>
    void inner_prod(XRange & x, YRange & y, ResultIterator result)
    {
      blas_traits< typename ::mgpu::range_traits<const XRange>::value_type,
                   typename ::mgpu::iterator_traits<ResultIterator>::value_type>
      ::template inner_prod<ResultIterator>(
        handle_,
        ::mgpu::range_traits<const XRange>::get_pointer(x),
        ::mgpu::range_traits<const YRange>::get_pointer(y),
        result,
        ::mgpu::range_traits<const XRange>::size(x),
        ::mgpu::range_traits<const XRange>::increment(x),
        ::mgpu::range_traits<const YRange>::increment(y));
    }

    /// calculate inner product (first vector is conjugated)
    template <typename XRange, typename YRange, typename ResultIterator>
    void inner_prod_c(XRange & x, YRange & y, ResultIterator result)
    {
      blas_traits< typename ::mgpu::range_traits<const XRange>::value_type,
                   typename ::mgpu::iterator_traits<ResultIterator>::value_type>
      ::template inner_prod_c<ResultIterator>(
        handle_,
        ::mgpu::range_traits<const XRange>::get_pointer(x),
        ::mgpu::range_traits<const YRange>::get_pointer(y),
        result,
        ::mgpu::range_traits<const XRange>::size(x),
        ::mgpu::range_traits<const XRange>::increment(x),
        ::mgpu::range_traits<const YRange>::increment(y));
    }

    /// calculate y = a*x + y
    template <typename AlphaIterator, typename XRange, typename YRange>
    void axpy(AlphaIterator alpha, const XRange & x, YRange & y)
    {
      blas_traits< typename ::mgpu::range_traits<const XRange>::value_type,
                   typename ::mgpu::iterator_traits<AlphaIterator>::value_type>
      ::template axpy<AlphaIterator>(
        handle_,
        alpha,
        ::mgpu::range_traits<const XRange>::get_pointer(x),
        ::mgpu::range_traits<YRange>::get_pointer(y),
        ::mgpu::range_traits<const XRange>::size(x),
        ::mgpu::range_traits<const XRange>::increment(x),
        ::mgpu::range_traits<YRange>::increment(y)
      );
    }

  private:

    /// the cuda blas handle
    cublasHandle_t handle_;
};


} // namespace cuda

} // namespace backend_detail

} // namespace mgpu

#endif // MGPU_BACKEND_CUDA_BLAS_BLAS_HPP
