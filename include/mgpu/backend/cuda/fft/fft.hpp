// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_FFT_FFT_HPP
#define MGPU_BACKEND_CUDA_FFT_FFT_HPP

/**
 * @file blas.hpp
 *
 * This header provides the mighty fft class
 */

#include <cuda_runtime.h>
#include <cufft.h>


#include <mgpu/core/dev_ptr.hpp>
#include <mgpu/container/range_traits.hpp>
#include <mgpu/backend/cuda/fft/fft_traits.hpp>
#include <mgpu/backend/cuda/fft/exception.hpp>
#include <mgpu/backend/dev_stream.hpp>

namespace mgpu
{

namespace backend_detail
{

namespace cuda
{


/**
 * @brief the fft class wraps all fft functionality
 */
template <typename InputType, typename OutputType>
class fft
{
  public:

    static const bool forward_possible  =
      fft_traits<InputType, OutputType>::forward_possible::value;

    static const bool inverse_possible  =
      fft_traits<InputType, OutputType>::inverse_possible::value;

    /// create fft object for 2D FFT
    inline fft(const std::size_t & dim1, const std::size_t & dim2,
      const std::size_t & batch = 1)
    {
      allocate(dim1, dim2, batch);
    }

    /// create empty fft object
    inline fft()
    { }

    /// explicitly allocate fft handle
    inline void allocate(const std::size_t & dim1, const std::size_t & dim2,
      const std::size_t & batch = 1)
    {
      int size = dim1*dim2;
      int embed[] = {dim1*dim2, dim1};
      int dims[] = {dim1, dim1};

      MGPU_CUDA_FFT_CALL(
        cufftPlanMany(&handle_, 2, dims, embed, 1, size, embed, 1, size,
          (cufftType)fft_traits<InputType, OutputType>::id::value, batch));
    }

    /// calculate forward FFT
    template <typename InputRange, typename OutputRange>
    void forward(InputRange & in, OutputRange & out)
    {
      MGPU_CUDA_FFT_CALL(
        (fft_traits<InputType, OutputType>::forward(handle_,
          ::mgpu::range_traits<InputRange>::get_pointer(in),
          ::mgpu::range_traits<OutputRange>::get_pointer(out))));
    }

    /// calculate inverse FFT
    template <typename InputRange, typename OutputRange>
    void inverse(InputRange & in, OutputRange & out)
    {
      MGPU_CUDA_FFT_CALL(
        (fft_traits<InputType, OutputType>::inverse(handle_,
          ::mgpu::range_traits<InputRange>::get_pointer(in),
          ::mgpu::range_traits<OutputRange>::get_pointer(out))));
    }

    /// set stream
    inline void set_stream(backend::dev_stream const & stream)
    { MGPU_CUDA_FFT_CALL(cufftSetStream(handle_, stream.get())); }

    /// reset stream
    inline void reset_stream()
    { MGPU_CUDA_FFT_CALL(cufftSetStream(handle_, 0)); }

    /// destroy the blas object
    inline ~fft() { MGPU_CUDA_FFT_CALL(cufftDestroy(handle_)); }

  private:
    cufftHandle handle_;
};


} // namespace cuda

} // namespace backend_detail

} // namespace mgpu

#endif // MGPU_BACKEND_CUDA_FFT_FFT_HPP
