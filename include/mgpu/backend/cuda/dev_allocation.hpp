// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_DEV_ALLOCATION_HPP
#define MGPU_BACKEND_CUDA_DEV_ALLOCATION_HPP

/**
 * @file dev_allocation.hpp
 *
 * This header provides device memory allocation functions for the CUDA
 * backend.
 */

#include <stddef.h>
#include <cuda_runtime.h>
#include <mgpu/core/dev_id.hpp>
#include <mgpu/backend/cuda/cuda_call.hpp>

namespace mgpu
{

template <typename T> class dev_ptr;

} // namespace mgpu

namespace mgpu
{
namespace backend_detail
{
namespace cuda
{

/**
 * @brief Allocate memory on device
 *
 * Allocate sizeof(T) * size bytes of memory on a CUDA device. The allocated
 * memory is suitably aligned for any kind of variable. The memory is not
 * cleared.
 *
 * @tparam T type that should be allocated
 *
 * @param size number of T's that should be allocated
 *
 * @return dev_ptr<T> identifying allocated device memory
 *
 * @ingroup backend
 */
template <typename T>
inline mgpu::dev_ptr<T> dev_malloc(const std::size_t & size)
{
  T * ptr;
  MGPU_CUDA_CALL(cudaMalloc((void**)&ptr, sizeof(T) * size));
  int dev_id;
  MGPU_CUDA_CALL(cudaGetDevice(&dev_id));
  return mgpu::dev_ptr<T>(ptr, dev_id);
}

/**
 * @brief Free memory on device
 *
 * Frees the memory space identified by ptr. Invalidates ptr.
 *
 * @param ptr device pointer identifying memory that should be freed
 *
 * @ingroup backend
 */
template <typename T>
inline void dev_free(mgpu::dev_ptr<T> & ptr)
{
  MGPU_CUDA_CALL(cudaFree(ptr.get_raw_pointer()));
  ptr.set_null();
}

/**
 * @brief Set memory on device
 *
 * Sets the memory space identified by ptr. to byte value
 *
 * @param ptr device pointer identifying memory that should be set
 *
 * @param value byte value the memory area should be filled with
 *
 * @param size number of T's that should be set to the value
 *
 * @ingroup backend
 */
template <typename T>
inline void dev_set(mgpu::dev_ptr<T> & ptr, int value,
  const std::size_t & size)
{
  MGPU_CUDA_CALL(cudaMemset(ptr.get_raw_pointer(), value, size*sizeof(T)));
}

/**
 * @brief Get id of device pointer points to

 * @param ptr device pointer
 *
 * @return id of device the pointer points to
 *
 * @ingroup backend
 */
template <typename T>
inline dev_id_t dev_id(const mgpu::dev_ptr<T> & ptr)
{
#ifdef MGPU_CUDA_SUPPORT_UNIFIED_ADDRESSING
  cudaPointerAttributes attributes;
  MGPU_CUDA_CALL(cudaPointerGetAttributes(&attributes,
    const_cast<T*>(ptr.get_raw_pointer())));
  return attributes.device;
#else
  BOOST_ASSERT_MSG(false,
    "dev_id not supported if unified addressing is disabled");
  return 0;
#endif
}

/**
 * @brief Get id of device pointer points to

 * @param ptr device pointer
 *
 * @return id of device the pointer points to
 *
 * @ingroup backend
 */
template <typename T>
inline dev_id_t dev_id(T * const & ptr)
{
  cudaPointerAttributes attributes;
  MGPU_CUDA_CALL(cudaPointerGetAttributes(&attributes, const_cast<T*>(ptr)));
  return attributes.device;
}

} // namespace cuda

} // namespace backend_detail

} // namespace mgpu




#endif // MGPU_BACKEND_CUDA_DEV_ALLOCATION_HPP

