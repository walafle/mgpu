// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_TRANSFER_BROADCAST_HPP
#define MGPU_TRANSFER_BROADCAST_HPP

/**
 * @file broadcast.hpp
 *
 * Provides various broadcast functions
 */

#include <mgpu/invoke.hpp>
#include <mgpu/backend/backend.hpp>
#include <mgpu/container/range_traits.hpp>
#include <mgpu/container/iterator_traits.hpp>
#include <mgpu/seg_dev_stream.hpp>
#include <mgpu/transfer/copy.hpp>
#include <mgpu/container/make_range.hpp>

namespace mgpu
{

/**
 * @brief broadcast from an input range to a segmented output iterator
 */
template <typename InputRange, typename OutputIterator>
OutputIterator broadcast(const InputRange & input, OutputIterator output,
  seg_dev_stream const & stream = default_seg_stream)
{
  // start a copy for each segment
  std::size_t segments =
    ::mgpu::iterator_traits<OutputIterator>::segments(output);
  for(std::size_t segment=0; segment<segments; segment++)
  {
    dev_rank_t rank =
      mgpu::iterator_traits<OutputIterator>::rank(output, segment);
    invoke(
      copy<
          typename ::mgpu::range_traits<const InputRange>::make_range_type
        , typename ::mgpu::iterator_traits<OutputIterator>::local_iterator
        , backend::dev_stream
        >
      , make_range(input)
      , mgpu::iterator_traits<OutputIterator>::begin_local(output, segment)
      , stream[rank]
      , rank
    );
  }
  return output;
}

} // namespace mgpu


#endif // MGPU_TRANSFER_BROADCAST_HPP
