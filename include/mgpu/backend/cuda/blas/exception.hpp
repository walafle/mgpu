// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_BLAS_EXCEPTION_HPP
#define MGPU_BACKEND_CUDA_BLAS_EXCEPTION_HPP

/**
 * @file exception.hpp
 *
 * This header provides exception functionality for the CUDA blas library
 */

#include <mgpu/backend/cuda/dev_exception.hpp>

namespace mgpu
{

namespace backend_detail
{

namespace cuda
{


// get the CUBLAS error string for exceptions
inline const char * blas_get_error_string(cublasStatus_t error)
{
  switch(error)
  {
    case CUBLAS_STATUS_SUCCESS:
    {
      return "the operation completed successfully";
    }
    case CUBLAS_STATUS_NOT_INITIALIZED:
    {
      return "the library was not initialized";
    }
    case CUBLAS_STATUS_ALLOC_FAILED:
    {
      return "the resource allocation failed";
    }
    case CUBLAS_STATUS_INVALID_VALUE:
    {
      return "an invalid numerical value was used as an argument";
    }
    case CUBLAS_STATUS_ARCH_MISMATCH:
    {
      return "an absent device architectural feature is required";
    }
    case CUBLAS_STATUS_MAPPING_ERROR:
    {
      return "an access to GPU memory space failed";
    }
    case CUBLAS_STATUS_EXECUTION_FAILED:
    {
      return "the GPU program failed to execute";
    }
    case CUBLAS_STATUS_INTERNAL_ERROR:
    {
      return "an internal operation failed";
    }
    default:
    {
      return "";
    }

  }
}


/**
 * @brief Safely call a CUDA BLAS function
 *
 * Call expression and check if it returned an error, if it returned an error
 * throw device_exception and include CUDA related information (error number and
 * error code).
 *
 * Throws with enable_current_exception so it can be accessed using
 * boost::current_exception()
 *
 * @param expression CUDA BLAS expression that is called
 *
 * @ingroup backend
 */

#ifdef MGPU_DEBUG
#warning "compiling debug version of mgpu blas (cuda)!"
#define MGPU_CUDA_BLAS_CALL(call)                                              \
  cublasStatus_t mgpu_cuda_blas_call_err = call;                               \
  if (CUBLAS_STATUS_SUCCESS != mgpu_cuda_blas_call_err)                        \
  {                                                                            \
    printf("cublas error %d in %s (file: %s, line %d):\n %s",                  \
      mgpu_cuda_blas_call_err, BOOST_CURRENT_FUNCTION, __FILE__, __LINE__,     \
      mgpu::backend_detail::cuda::                                             \
        blas_get_error_string(mgpu_cuda_blas_call_err));                       \
    throw boost::enable_current_exception(                                     \
        mgpu::backend_detail::cuda::device_exception()) <<                     \
      mgpu::backend_detail::cuda::cuda_err_code(mgpu_cuda_blas_call_err) <<    \
      mgpu::backend_detail::cuda::cuda_err_text(                               \
        mgpu::backend_detail::cuda::                                           \
          blas_get_error_string(mgpu_cuda_blas_call_err)) <<                   \
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

#define MGPU_CUDA_BLAS_CALL(call)                                              \
  cublasStatus_t mgpu_cuda_blas_call_err = call;                               \
  if (CUBLAS_STATUS_SUCCESS != mgpu_cuda_blas_call_err)                        \
  {                                                                            \
    throw boost::enable_current_exception(                                     \
        mgpu::backend_detail::cuda::device_exception()) <<                     \
      mgpu::backend_detail::cuda::cuda_err_code(mgpu_cuda_blas_call_err) <<    \
      mgpu::backend_detail::cuda::cuda_err_text(                               \
        mgpu::backend_detail::cuda::                                           \
          blas_get_error_string(mgpu_cuda_blas_call_err)) <<                   \
      boost::throw_function(BOOST_CURRENT_FUNCTION) <<                         \
      boost::throw_file(__FILE__) <<                                           \
      boost::throw_line((int)__LINE__);                                        \
  }                                                                            \
  /**/

#endif // MGPU_DEBUG

} // namespace cuda

} // namespace backend_detail

} // namespace mgpu

#endif // MGPU_BACKEND_CUDA_BLAS_EXCEPTION_HPP
