// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_CUDA_CALL_HPP
#define MGPU_BACKEND_CUDA_CUDA_CALL_HPP

/**
 * @file cuda_call.hpp
 *
 * This header provides macros for shorthand check of return values and
 * exception throwing with CUDA runtime calls.
 */

#include <stdio.h>

#include <mgpu/backend/cuda/dev_exception.hpp>

/**
 * @brief Safely call a CUDA API function
 *
 * Call expression and check if it returned an error, if it returned an error
 * throw device_exception and include CUDA related information (error number and
 * error code).
 *
 * Throws with enable_current_exception so it can be accessed using
 * boost::current_exception()
 *
 * @param expression CUDA API expression that is called
 *
 * @ingroup backend
 */

#ifdef MGPU_DEBUG
#warning "compiling debug version of mgpu backend call (cuda)!"
#define MGPU_CUDA_CALL(call)                                                   \
  {                                                                            \
    cudaError_t mgpu_cuda_call_err = call;                                     \
    if (cudaSuccess != mgpu_cuda_call_err)                                     \
    {                                                                          \
      printf("error %d in %s (file: %s, line %d):\n %s",                       \
        mgpu_cuda_call_err, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__,        \
        cudaGetErrorString(mgpu_cuda_call_err));                               \
      throw boost::enable_current_exception(                                   \
          mgpu::backend_detail::cuda::device_exception()) <<                   \
        mgpu::backend_detail::cuda::cuda_err_code(mgpu_cuda_call_err) <<       \
        mgpu::backend_detail::cuda::cuda_err_text(                             \
          cudaGetErrorString(mgpu_cuda_call_err)) <<                           \
        boost::throw_function(BOOST_CURRENT_FUNCTION) <<                       \
        boost::throw_file(__FILE__) <<                                         \
        boost::throw_line((int)__LINE__);                                      \
    }                                                                          \
    cudaError_t mgpu_cuda_call_err2 = cudaDeviceSynchronize();                 \
    if (cudaSuccess != mgpu_cuda_call_err2)                                    \
    {                                                                          \
      printf("error sync %d in %s (file: %s, line %d):\n %s",                  \
        mgpu_cuda_call_err2, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__,       \
        cudaGetErrorString(mgpu_cuda_call_err2));                              \
      throw boost::enable_current_exception(                                   \
          mgpu::backend_detail::cuda::device_exception()) <<                   \
        mgpu::backend_detail::cuda::cuda_err_code(mgpu_cuda_call_err2) <<      \
        mgpu::backend_detail::cuda::cuda_err_text(                             \
          cudaGetErrorString(mgpu_cuda_call_err2)) <<                          \
        boost::throw_function(BOOST_CURRENT_FUNCTION) <<                       \
        boost::throw_file(__FILE__) <<                                         \
        boost::throw_line((int)__LINE__);                                      \
    }                                                                          \
  }                                                                            \
  /**/

#else // MGPU_DEBUG

#define MGPU_CUDA_CALL(call)                                                   \
  {                                                                            \
    cudaError_t mgpu_cuda_call_err = call;                                     \
    if (cudaSuccess != mgpu_cuda_call_err)                                     \
    {                                                                          \
      throw boost::enable_current_exception(                                   \
          mgpu::backend_detail::cuda::device_exception()) <<                  \
        mgpu::backend_detail::cuda::cuda_err_code(mgpu_cuda_call_err) <<      \
        mgpu::backend_detail::cuda::cuda_err_text(                            \
          cudaGetErrorString(mgpu_cuda_call_err)) <<                           \
        boost::throw_function(BOOST_CURRENT_FUNCTION) <<                       \
        boost::throw_file(__FILE__) <<                                         \
        boost::throw_line((int)__LINE__);                                      \
    }                                                                          \
  }                                                                            \
  /**/

#endif // MGPU_DEBUG


#endif // MGPU_BACKEND_CUDA_CUDA_CALL_HPP
