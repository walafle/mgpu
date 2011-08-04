// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_TRANSFER_COPY_HPP
#define MGPU_TRANSFER_COPY_HPP

/** 
 * @file copy.hpp
 *
 * Provides various copy functions
 */

#include <mgpu/backend/transfer.hpp>
#include <mgpu/backend/dev_stream.hpp>
#include <mgpu/container/make_range.hpp>
#include <mgpu/container/range_traits.hpp>
#include <mgpu/container/iterator_traits.hpp>
#include <mgpu/transfer/detail/copy.hpp>

namespace mgpu
{

/**
 * @brief synchronously copy from an input range to an output iterator
 */
template <typename InputRange, typename OutputIterator>
OutputIterator copy(const InputRange & input, OutputIterator output)
{
  detail::copy<InputRange, OutputIterator>(input, output,
    typename ::mgpu::range_traits<InputRange>::segmented_tag(),
    typename ::mgpu::iterator_traits<OutputIterator>::segmented_tag(),
    typename ::mgpu::range_traits<InputRange>::location_tag(),
    typename ::mgpu::iterator_traits<OutputIterator>::location_tag());
  return output;
}

/**
 * @brief asynchronously copy from an input range to an output iterator
 */
template <typename InputRange, typename OutputIterator, typename StreamType>
OutputIterator copy(const InputRange & input, OutputIterator output,
  StreamType const & stream)
{
  detail::copy<InputRange, OutputIterator>(input, output,
    typename ::mgpu::range_traits<InputRange>::segmented_tag(),
    typename ::mgpu::iterator_traits<OutputIterator>::segmented_tag(),
    typename ::mgpu::range_traits<InputRange>::location_tag(),
    typename ::mgpu::iterator_traits<OutputIterator>::location_tag(),
    stream);
  return output;
}

} // namespace mgpu


#endif // MGPU_TRANSFER_COPY_HPP
