// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BLAS_DETAIL_BLAS_IPP
#define MGPU_BLAS_DETAIL_BLAS_IPP

/**
 * @file blas.ipp
 *
 * This header contains the blas implementation
 */

#include <mgpu/blas/detail/blas_async.ipp>
#include <mgpu/core/ref.hpp>
#include <mgpu/invoke.hpp>


namespace mgpu
{

// allocate segmented blas object -----

blas::blas(const rank_group & ranks) :
  ranks_(ranks), segments_(ranks.size())
{
  // allocate blas objects
  for(unsigned int segment=0; segment<segments_; segment++)
  {
    // construct blas object with deferred allocation
    resources_[segment] = new backend::blas();

    // trigger allocation
    invoke(
        &backend::blas::allocate
      , resources_[segment]
      , ranks_[segment]
      );
  }
}


// destroy segmented blas object -----

blas::~blas()
{
  // free blas objects
  for(unsigned int segment=0; segment<segments_; segment++)
  {
    // trigger allocation
    invoke(detail::free_single_blas_impl,
      resources_[segment], ranks_[segment]);
  }
}

// set parameter passing type (on device or on host) -----

void blas::set_scalar_device()
{
  for(unsigned int segment=0; segment<segments_; segment++)
  {
    invoke(
        &backend::blas::set_scalar_device
      , resources_[segment]
      , ranks_[segment]
      );
  }
}

void blas::set_scalar_host()
{
  for(unsigned int segment=0; segment<segments_; segment++)
  {
    invoke(
        &backend::blas::set_scalar_host
      , resources_[segment]
      , ranks_[segment]
      );
  }
}


// axpy -----

template <typename AlphaIterator, typename XRange, typename YRange>
void blas::axpy_impl(AlphaIterator alpha, const XRange & x,
  YRange & y, device_memory_tag)
{
  std::size_t segments = ::mgpu::range_traits<XRange>::segments(x);

  for(std::size_t segment=0; segment<segments; segment++)
  {
    invoke(
      &backend::blas::axpy<
          typename ::mgpu::iterator_traits<AlphaIterator>::local_iterator,
          typename ::mgpu::range_traits<XRange>::local_range,
          typename ::mgpu::range_traits<YRange>::local_range
        >,
      resources_[segment],
      ::mgpu::iterator_traits<AlphaIterator>::begin_local(alpha, segment),
      make_range(x, segment),
      make_range(y, segment),
      ranks_[segment]
    );
  }
}

template <typename AlphaIterator, typename XRange, typename YRange>
void blas::axpy_impl(AlphaIterator alpha, const XRange & x, YRange & y,
  host_memory_tag)
{
  std::size_t segments = ::mgpu::range_traits<XRange>::segments(x);

  for(std::size_t segment=0; segment<segments; segment++)
  {
    invoke(
      &backend::blas::axpy<
          AlphaIterator,
          typename ::mgpu::range_traits<XRange>::local_range,
          typename ::mgpu::range_traits<YRange>::local_range
        >,
      resources_[segment],
      alpha,
      make_range(x, segment),
      make_range(y, segment),
      ranks_[segment]
    );
  }
}


// inner_prod -----

template <typename XRange, typename YRange, typename ResultIterator>
void blas::inner_prod_impl(XRange & x, YRange & y, ResultIterator result,
  is_segmented_tag)
{
  std::size_t segments = ::mgpu::range_traits<XRange>::segments(x);

  for(std::size_t segment=0; segment<segments; segment++)
  {
    invoke(
      &backend::blas::inner_prod<
        typename ::mgpu::range_traits<XRange>::local_range,
        typename ::mgpu::range_traits<YRange>::local_range,
        typename ::mgpu::iterator_traits<ResultIterator>::local_iterator
        >,
      resources_[segment],
      make_range(x, segment),
      make_range(y, segment),
      ::mgpu::iterator_traits<ResultIterator>::begin_local(result, segment),
      ranks_[segment]
    );
  }
}

template <typename XRange, typename YRange, typename ResultIterator>
void blas::inner_prod_impl(XRange & x, YRange & y, ResultIterator result,
  is_not_segmented_tag)
{
  std::size_t segments = ::mgpu::range_traits<XRange>::segments(x);

  for(std::size_t segment=0; segment<segments; segment++)
  {
    invoke(
      &backend::blas::inner_prod<
        typename ::mgpu::range_traits<XRange>::local_range,
        typename ::mgpu::range_traits<YRange>::local_range,
        ResultIterator
        >,
      resources_[segment],
      make_range(x, segment),
      make_range(y, segment),
      result+segment,
      ranks_[segment]
    );
  }
}


// inner_prod_c -----

template <typename XRange, typename YRange, typename ResultIterator>
void blas::inner_prod_c_impl(XRange & x, YRange & y, ResultIterator result,
  is_segmented_tag)
{
  std::size_t segments = ::mgpu::range_traits<XRange>::segments(x);

  for(std::size_t segment=0; segment<segments; segment++)
  {
    invoke(
      &backend::blas::inner_prod_c<
        typename ::mgpu::range_traits<XRange>::local_range,
        typename ::mgpu::range_traits<YRange>::local_range,
        typename ::mgpu::iterator_traits<ResultIterator>::local_iterator
        >,
      resources_[segment],
      make_range(x, segment),
      make_range(y, segment),
      ::mgpu::iterator_traits<ResultIterator>::begin_local(result, segment),
      ranks_[segment]
    );
  }
}

template <typename XRange, typename YRange, typename ResultIterator>
void blas::inner_prod_c_impl(XRange & x, YRange & y, ResultIterator result,
  is_not_segmented_tag)
{
  std::size_t segments = ::mgpu::range_traits<XRange>::segments(x);

  for(std::size_t segment=0; segment<segments; segment++)
  {
    invoke(
      &backend::blas::inner_prod_c<
        typename ::mgpu::range_traits<XRange>::local_range,
        typename ::mgpu::range_traits<YRange>::local_range,
        ResultIterator
        >,
      resources_[segment],
      make_range(x, segment),
      make_range(y, segment),
      result+segment,
      ranks_[segment]
    );
  }
}

} // namespace mgpu



#endif // MGPU_BLAS_DETAIL_BLAS_IPP
