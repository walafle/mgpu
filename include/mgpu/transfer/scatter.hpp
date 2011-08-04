// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_TRANSFER_SCATTER_HPP
#define MGPU_TRANSFER_SCATTER_HPP

/**
 * @file scatter.hpp
 *
 * Provides various scatter functions
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
 * @brief scatter from an input range to a segmented output iterator
 */
template <typename InputRange, typename OutputIterator>
OutputIterator scatter(const InputRange & input, OutputIterator output,
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


#endif // MGPU_TRANSFER_SCATTER_HPP
