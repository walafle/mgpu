// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_FFT_DETAIL_FFT_ASYNC_IPP
#define MGPU_FFT_DETAIL_FFT_ASYNC_IPP

/**
 * @file fft_async.ipp
 *
 * This header contains the segmented fft asynchronous functions
 */

#include <mgpu/backend/fft.hpp>

namespace mgpu
{

namespace detail
{

// asynchronous free methods
// _____________________________________________________________________________

template <typename InputType, typename OutputType>
inline void free_single_fft_impl(backend::fft<InputType, OutputType> * f)
{
  // delete the handle
  if(f != NULL)
  {
    delete f;
  }
}

} // namespace detail

} // namespace mgpu



#endif // MGPU_FFT_DETAIL_FFT_ASYNC_IPP
