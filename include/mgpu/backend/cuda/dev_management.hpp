// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_DEV_MANAGEMENT_HPP
#define MGPU_BACKEND_CUDA_DEV_MANAGEMENT_HPP

/**
 * @file utils.hpp
 *
 * This header provides various utility wrappers around CUDA runtime API calls.
 */

#include <utility>

#include <cuda_runtime.h>

#include <mgpu/core/dev_id.hpp>
#include <mgpu/backend/cuda/cuda_call.hpp>
#include <mgpu/backend/cuda/dev_stream.hpp>

namespace mgpu
{
namespace backend_detail
{
namespace cuda
{

/**
 * @brief Return number of devices
 *
 * Return the number of devices with compute capability greater or equal to 1.0
 * that are available for execution. If only device emulation mode is available,
 * 0 is returned.
 *
 * @return the number of devices in the system, 0 if no devices are found
 *
 * @ingroup backend
 */
inline unsigned int get_dev_count()
{
  int devicecount;
  MGPU_CUDA_CALL(cudaGetDeviceCount(&devicecount));

  // only one device found, check if it is the emulator
  if(devicecount == 1)
  {
    cudaDeviceProp prop;
    MGPU_CUDA_CALL(cudaGetDeviceProperties(&prop, 0));
    if(prop.major == 9999 && prop.minor == 9999)
    {
      devicecount = 0;
    }
  }
  return devicecount;
}


/**
 * @brief Set device as the device on which the active host thread executes the
 * device code.
 *
 * @ingroup backend
 */
inline void set_dev(const dev_id_t device)
{
  MGPU_CUDA_CALL(cudaSetDevice(device));
}

/**
 * @brief Set device as the device on which the active host thread executes the
 * device code.
 *
 * @ingroup backend
 */
inline dev_id_t get_dev()
{
  int device;
  MGPU_CUDA_CALL(cudaGetDevice(&device));
  return device;
}

/**
 * @brief Block until the current device has completed all preceding requested
 * tasks.
 *
 * @ingroup backend
 */
inline void sync_dev()
{
  MGPU_CUDA_CALL(cudaDeviceSynchronize());
}

/**
 * @brief Block until the current device has completed all preceding requested
 * tasks in the specified stream.
 *
 * @param stream stream that should be synchronized
 *
 * @ingroup backend
 */
inline void sync_dev(const dev_stream & stream)
{
  MGPU_CUDA_CALL(cudaStreamSynchronize(stream.get()));
}

/**
 * @brief Clean up and destroy all resources associated with the current device.
 *
 * @ingroup backend
 */
inline void reset_dev()
{
  MGPU_CUDA_CALL(cudaDeviceReset());
}

/**
 * @brief Allow current device to access memory on remote device
 *
 * @param to enable access to this device memory
 *
 * @ingroup backend
 */
inline void enable_p2p(dev_id_t to)
{
  MGPU_CUDA_CALL(cudaDeviceEnablePeerAccess(to, 0));
}

/**
 * @brief Disallow current device to access memory on remote device
 *
 * @param to disable access to this device memory
 *
 * @ingroup backend
 */
inline void disable_p2p(dev_id_t to)
{
  MGPU_CUDA_CALL(cudaDeviceDisablePeerAccess(to));
}

/**
 * @brief Check if remote memory access to device is possible from other device
 *
 * @param from memory access from this device
 * @param to memory access to this device
 *
 * @ingroup backend
 */
inline bool p2p_possible(dev_id_t from, dev_id_t to)
{
  int can_access;
  MGPU_CUDA_CALL(cudaDeviceCanAccessPeer(&can_access, from, to));
  return can_access;
}

/**
 * @brief Returns free amount of memory available for allocation by the device
 * in bytes
 */
inline std::size_t get_free_mem()
{
  size_t free;
  size_t total;
  MGPU_CUDA_CALL(cudaMemGetInfo(&free, &total));
  return free;
}

/**
 * @brief Returns total amount of memory available for allocation by the device
 * in bytes
 */
inline std::size_t get_total_mem()
{
  size_t free;
  size_t total;
  MGPU_CUDA_CALL(cudaMemGetInfo(&free, &total));
  return total;
}

/**
 * @brief Returns the compute capability of the device (major and minor)
 */
inline std::pair<int, int> get_compute_capability(dev_id_t id)
{
  cudaDeviceProp properties;
  MGPU_CUDA_CALL(cudaGetDeviceProperties(&properties, id));
  return std::pair<int, int>(properties.major, properties.minor);
}

} // namespace cuda

} // namespace backend_detail

} // namespace mgpu



#endif // MGPU_BACKEND_CUDA_DEV_MANAGEMENT_HPP
