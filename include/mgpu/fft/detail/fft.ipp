// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_FFT_DETAIL_FFT_IPP
#define MGPU_FFT_DETAIL_FFT_IPP

/**
 * @file fft.ipp
 *
 * This header provides the implementation of the mighty fft class
 */

#include <mgpu/invoke.hpp>
#include <mgpu/backend/fft.hpp>
#include <mgpu/fft/detail/fft_async.ipp>
#include <mgpu/core/detail/splitter.hpp>
#include <mgpu/container/range_traits.hpp>
#include <mgpu/container/make_range.hpp>

namespace mgpu
{

// destroy segmented blas object -----

template <typename InputType, typename OutputType>
fft<InputType,OutputType>::fft(std::size_t dim1, std::size_t dim2,
  std::size_t batch, const rank_group & ranks) :
  ranks_(ranks), batch_(batch), segments_(ranks.size())
{
  try
  {
    // use splitter to split the batches across segments
    detail::splitter s(batch_, 1, segments_);

    // allocate fft objects
    for(unsigned int segment=0; segment<segments_; segment++)
    {
      blocks_[segment] = s++;
      if(blocks_[segment] < 1) continue;

      // construct fft object with deferred allocation
      resources_[segment] = new backend::fft<InputType, OutputType>();

      // trigger allocation
      invoke(
          &backend::fft<InputType, OutputType>::allocate
        , resources_[segment]
        , dim1
        , dim2
        , blocks_[segment]
        , ranks_[segment]
        );
    }
  }
  catch(mgpu_exception e)
  {
    throw e;
  }
}

// set stream -----

template <typename InputType, typename OutputType>
void fft<InputType,OutputType>::set_stream(seg_dev_stream & stream)
{
  for(unsigned int segment=0; segment<segments_; segment++)
  {
    // trigger set stream
    invoke(
        &backend::fft<InputType, OutputType>::set_stream
      , resources_[segment]
      , stream[segment]
      , ranks_[segment]
      );
  }
}

// reset stream -----

template <typename InputType, typename OutputType>
void fft<InputType,OutputType>::reset_stream()
{
  for(unsigned int segment=0; segment<segments_; segment++)
  {
    // trigger reset stream
    invoke(
        &backend::fft<InputType, OutputType>::reset_stream
      , resources_[segment]
      , ranks_[segment]
      );
  }
}

// destroy blas object -----

template <typename InputType, typename OutputType>
fft<InputType,OutputType>::~fft()
{
  // free blas objects
  for(unsigned int segment=0; segment<segments_; segment++)
  {
    if(blocks_[segment] < 1) continue;

    // trigger free
    invoke(detail::free_single_fft_impl<InputType, OutputType>,
      resources_[segment], ranks_[segment]);
  }
}

// forward transform -----

template <typename InputType, typename OutputType>
template <typename InputRange, typename OutputRange>
void fft<InputType, OutputType>::forward_impl(
  InputRange & in, OutputRange & out)
{
  typedef typename ::mgpu::range_traits<InputRange>::make_range_type R1;
  typedef typename ::mgpu::range_traits<OutputRange>::make_range_type R2;

  for(unsigned int segment=0; segment<segments_; segment++)
  {
    if(blocks_[segment] < 1) continue;

    // trigger transform
    invoke(
        &backend::fft<InputType, OutputType>::template forward<R1, R2>
      , resources_[segment]
      , make_range(in, segment)
      , make_range(out, segment)
      , ranks_[segment]
      );
  }
}


// inverse transform -----

template <typename InputType, typename OutputType>
template <typename InputRange, typename OutputRange>
void fft<InputType, OutputType>::inverse_impl(
  InputRange & in, OutputRange & out)
{
  typedef typename ::mgpu::range_traits<InputRange>::make_range_type R1;
  typedef typename ::mgpu::range_traits<OutputRange>::make_range_type R2;

  for(unsigned int segment=0; segment<segments_; segment++)
  {
    if(blocks_[segment] < 1) continue;

    // trigger transform
    invoke(
        &backend::fft<InputType, OutputType>::template inverse<R1, R2>
      , resources_[segment]
      , make_range(in, segment)
      , make_range(out, segment)
      , ranks_[segment]
      );
  }
}

} // namespace mgpu


#endif // MGPU_FFT_DETAIL_FFT_IPP
