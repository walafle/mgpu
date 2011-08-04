// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_FFT_EXCEPTION_HPP
#define MGPU_BACKEND_CUDA_FFT_EXCEPTION_HPP

/**
 * @file exception.hpp
 *
 * This header provides exception functionality for the CUDA FFT library
 */

#include <mgpu/backend/cuda/dev_exception.hpp>

namespace mgpu
{

namespace backend_detail
{

namespace cuda
{

// get the CUFFT error string for exceptions
inline const char * fft_get_error_string(cufftResult_t error)
{
  switch(error)
  {
    case CUFFT_SUCCESS:
    {
      return "Any CUFFT operation is successful.";
    }
    case CUFFT_INVALID_PLAN:
    {
      return "CUFFT is passed an invalid plan handle.";
    }
    case CUFFT_ALLOC_FAILED:
    {
      return "CUFFT failed to allocate GPU memory.";
    }
    case CUFFT_INVALID_TYPE:
    {
      return "The user requests an unsupported type.";
    }
    case CUFFT_INVALID_VALUE:
    {
      return "The user specifies a bad memory pointer.";
    }
    case CUFFT_INTERNAL_ERROR:
    {
      return "Used for all internal driver errors.";
    }
    case CUFFT_EXEC_FAILED:
    {
      return "CUFFT failed to execute an FFT on the GPU.";
    }
    case CUFFT_SETUP_FAILED:
    {
      return "The CUFFT library failed to initialize.";
    }
    case CUFFT_INVALID_SIZE:
    {
      return "The user specifies an unsupported FFT size.";
    }
    case CUFFT_UNALIGNED_DATA:
    {
      return "Input or output does not satisfy texture alignment requirements.";
    }
    default:
    {
      return "";
    }
  }
}


/**
 * @brief Safely call a CUDA FFT function
 *
 * Call expression and check if it returned an error, if it returned an error
 * throw device_exception and include CUDA related information (error number and
 * error code).
 *
 * Throws with enable_current_exception so it can be accessed using
 * boost::current_exception()
 *
 * @param expression CUDA FFT expression that is called
 *
 * @ingroup platform
 */

#ifdef MGPU_DEBUG
#warning "compiling debug version of mgpu fft (cuda)!"
#define MGPU_CUDA_FFT_CALL(call)                                               \
  cufftResult mgpu_cuda_fft_call_err = call;                                   \
  if (CUFFT_SUCCESS != mgpu_cuda_fft_call_err)                                 \
  {                                                                            \
    printf("cufft error %d in %s (file: %s, line %d):\n %s",                   \
      mgpu_cuda_fft_call_err, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__,      \
      mgpu::backend_detail::cuda::                                             \
        fft_get_error_string(mgpu_cuda_fft_call_err));                         \
    throw boost::enable_current_exception(                                     \
        mgpu::backend_detail::cuda::device_exception()) <<                     \
      mgpu::backend_detail::cuda::cuda_err_code(mgpu_cuda_fft_call_err) <<     \
      mgpu::backend_detail::cuda::cuda_err_text(                               \
        mgpu::backend_detail::cuda::                                           \
          fft_get_error_string(mgpu_cuda_fft_call_err)) <<                     \
      boost::throw_function(BOOST_CURRENT_FUNCTION) <<                         \
      boost::throw_file(__FILE__) <<                                           \
      boost::throw_line((int)__LINE__);                                        \
  }                                                                            \
  cudaError_t mgpu_cuda_call_err2 = cudaDeviceSynchronize();                   \
  if (cudaSuccess != mgpu_cuda_call_err2)                                      \
  {                                                                            \
    printf("error sync %d in %s (file: %s, line %d):\n %s",                    \
      mgpu_cuda_call_err2, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__,         \
      cudaGetErrorString(mgpu_cuda_call_err2));                                \
    throw boost::enable_current_exception(                                     \
        mgpu::backend_detail::cuda::device_exception()) <<                     \
      mgpu::backend_detail::cuda::cuda_err_code(mgpu_cuda_call_err2) <<        \
      mgpu::backend_detail::cuda::cuda_err_text(                               \
        cudaGetErrorString(mgpu_cuda_call_err2)) <<                            \
      boost::throw_function(BOOST_CURRENT_FUNCTION) <<                         \
      boost::throw_file(__FILE__) <<                                           \
      boost::throw_line((int)__LINE__);                                        \
  }                                                                            \
  /**/

#else // MGPU_DEBUG

#define MGPU_CUDA_FFT_CALL(call)                                               \
  cufftResult mgpu_cuda_fft_call_err = call;                                   \
  if (CUFFT_SUCCESS != mgpu_cuda_fft_call_err)                                 \
  {                                                                            \
    throw boost::enable_current_exception(                                     \
        mgpu::backend_detail::cuda::device_exception()) <<                     \
      mgpu::backend_detail::cuda::cuda_err_code(mgpu_cuda_fft_call_err) <<     \
      mgpu::backend_detail::cuda::cuda_err_text(                               \
        mgpu::backend_detail::cuda::                                           \
          fft_get_error_string(mgpu_cuda_fft_call_err)) <<                     \
      boost::throw_function(BOOST_CURRENT_FUNCTION) <<                         \
      boost::throw_file(__FILE__) <<                                           \
      boost::throw_line((int)__LINE__);                                        \
  }                                                                            \
  /**/

#endif // MGPU_DEBUG


} // namespace cuda

} // namespace backend_detail

} // namespace mgpu

#endif // MGPU_BACKEND_CUDA_FFT_EXCEPTION_HPP
