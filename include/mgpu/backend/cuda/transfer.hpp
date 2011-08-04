// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_TRANSFER_HPP
#define MGPU_BACKEND_CUDA_TRANSFER_HPP

/**
 * @file transfer.hpp
 *
 * This header provides transfer functions between device memory and host memory
 * or memory copies on device memory
 */
#include <stddef.h>

#include <boost/mpl/int.hpp>
#include <cuda_runtime.h>

#include <mgpu/core/2d.hpp>
#include <mgpu/core/dev_id.hpp>
#include <mgpu/core/dev_set_scoped.hpp>
#include <mgpu/container/tags.hpp>
#include <mgpu/backend/cuda/cuda_call.hpp>
#include <mgpu/backend/cuda/dev_management.hpp>

namespace mgpu
{
  template <typename T> class dev_ptr;
}

namespace mgpu
{
namespace backend_detail
{
namespace cuda
{



// copy data to device _________________________________________________________

/**
 * @brief Copy data to device asynchronously
 *
 * Copy count Ts to the device
 *
 * @tparam T type that should be copied
 *
 * @param dst memory copy destination on device
 * @param src memory copy source in host memory
 * @param count number of Ts to copy
 * @param stream stream in which the copy should be performed
 *
 * @ingroup backend
 */
template<typename T>
void copy(const T * const & src, dev_ptr<T> dst, std::size_t count,
  const dev_stream & stream = default_stream)
{
  dev_set_scoped dev_setter(dst.dev_id());
  MGPU_CUDA_CALL(cudaMemcpyAsync(dst.get_raw_pointer(), src, sizeof(T)*count,
    cudaMemcpyHostToDevice, stream.get()));
}



// copy data from dev __________________________________________________________

/**
 * @brief Copy data from device asynchronously
 *
 * Copy count Ts from the device
 *
 * @tparam T type that should be copied
 *
 * @param dst memory copy destination in host memory
 * @param src memory copy source on device
 * @param count number of Ts to copy
 * @param stream stream in which the copy should be performed
 *
 * @ingroup backend
 */
template<typename T>
void copy(const dev_ptr<T> & src, T * dst, std::size_t count,
  const dev_stream & stream = default_stream)
{
  dev_set_scoped dev_setter(src.dev_id());
  MGPU_CUDA_CALL(cudaMemcpyAsync(dst, src.get_raw_pointer(), sizeof(T)*count,
    cudaMemcpyDeviceToHost, stream.get()));
}



// copy on device or between devices ___________________________________________

/**
 * @brief Copy data on device asynchronously
 *
 * Copy count Ts on the device
 *
 * @tparam T type of data that is copied
 *
 * @param dst memory copy destination on device
 * @param src memory copy source on device
 * @param count number of Ts to copy
 * @param stream stream in which the copy should be performed
 *
 * @ingroup backend
 */
template<typename T>
void copy(const dev_ptr<T> & src, dev_ptr<T> dst, std::size_t count,
  const dev_stream & stream = default_stream)
{
  dev_set_scoped dev_setter(src.dev_id());
  if(src.dev_id() != dst.dev_id())
  {
    MGPU_CUDA_CALL(cudaMemcpyPeerAsync(dst.get_raw_pointer(), dst.dev_id(),
      src.get_raw_pointer(), src.dev_id(), sizeof(T)*count, stream.get()));
  }
  else
  {
    MGPU_CUDA_CALL(cudaMemcpyAsync(dst.get_raw_pointer(), src.get_raw_pointer(),
      sizeof(T)*count, cudaMemcpyDeviceToDevice, stream.get()));
  }
}



} // namespace cuda

} // namespace backend_detail

} // namespace mgpu




#endif // MGPU_BACKEND_CUDA_TRANSFER_HPP

