// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_DEV_STREAM_HPP
#define MGPU_BACKEND_CUDA_DEV_STREAM_HPP

/**
 * @file dev_stream.hpp
 *
 * This header contains the dev_stream class.
 */

#include <cuda_runtime.h>
#include <mgpu/backend/cuda/cuda_call.hpp>


namespace mgpu
{
namespace backend_detail
{
namespace cuda
{

/**
 * @brief a device stream
 *
 * A sequence of operations that are guaranteed to execute in order. Operations
 * on different streams may overlap.
 */
struct dev_stream
{
  /**
   * create stream
   */
  inline dev_stream() { create(); }

  /**
   * create default stream
   */
  explicit inline dev_stream(int) : stream_(0) { }

  /**
   * create stream without actually creating it
   */
  explicit inline dev_stream(bool deferred_creation) : stream_(0)
  {
    if(!deferred_creation)
    {
      create();
    }
  }

  /**
   * destroy stream
   */
  inline ~dev_stream()
  { if(stream_ != 0) MGPU_CUDA_CALL(cudaStreamDestroy(stream_)); }

  /**
   * access the raw stream handle
   */
  inline cudaStream_t get() const { return stream_; }

  /// create stream
  inline void create() { MGPU_CUDA_CALL(cudaStreamCreate(&stream_)); }

  private:

    /// cuda stream
    cudaStream_t stream_;
};

extern const dev_stream default_stream;

} // namespace cuda

} // namespace backend_detail

} // namespace mgpu


#endif // MGPU_BACKEND_CUDA_DEV_STREAM_HPP

