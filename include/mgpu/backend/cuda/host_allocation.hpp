// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_HOST_ALLOCATION_HPP
#define MGPU_BACKEND_CUDA_HOST_ALLOCATION_HPP

/**
 * @file host_allocation.hpp
 *
 * This header provides host memory allocation functions for the CUDA
 * backend.
 */

#include <stddef.h>
#include <cuda_runtime.h>
#include <mgpu/backend/cuda/cuda_call.hpp>

namespace mgpu
{
namespace backend_detail
{
namespace cuda
{

/**
 * @brief Allocate memory on host for optimized host to device transfer
 *
 * This implementation uses cudaHostAlloc to allocate pinned memory.
 *
 * @param size numter of T's that should be allocated
 *
 * @return pointer to allocated host memory
 *
 * @ingroup backend
 */
template <typename T>
inline T * host_opt_malloc(const std::size_t & size)
{
  T * ptr;
  MGPU_CUDA_CALL(cudaHostAlloc(reinterpret_cast<void**>(&ptr),
    size * sizeof(T), cudaHostAllocPortable));
  return ptr;
}

/**
 * @brief Free memory on host that was allocated for optimized host to device
 * transfer
 *
 * This implementation uses cudaFreeHost to free pinned memory.
 *
 * @param ptr reference to the pointer that should be freed
 *
 * @ingroup backend
 */
template <typename T>
inline void host_opt_free(T *& ptr)
{
  MGPU_CUDA_CALL(cudaFreeHost(ptr));
}

} // namespace cuda

} // namespace backend_detail

} // namespace mgpu




#endif // MGPU_BACKEND_CUDA_HOST_ALLOCATION_HPP

