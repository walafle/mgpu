// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_TRANSFER_GATHER_HPP
#define MGPU_TRANSFER_GATHER_HPP

/**
 * @file gather.hpp
 *
 * Provides various gather functions
 */

#include <mgpu/invoke.hpp>
#include <mgpu/container/range_traits.hpp>
#include <mgpu/container/iterator_traits.hpp>
#include <mgpu/seg_dev_stream.hpp>
#include <mgpu/transfer/detail/copy.hpp>

namespace mgpu
{

/**
 * @brief gather from an segmented input range to an output iterator
 */
template <typename InputRange, typename OutputIterator>
OutputIterator gather(const InputRange & input, OutputIterator output,
  seg_dev_stream const & stream = default_seg_stream)
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


#endif // MGPU_TRANSFER_GATHER_HPP
