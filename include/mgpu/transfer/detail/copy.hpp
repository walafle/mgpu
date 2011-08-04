// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_TRANSFER_DETAIL_COPY_HPP
#define MGPU_TRANSFER_DETAIL_COPY_HPP

/**
 * @file copy.hpp
 *
 * Provides implementation of copy functions
 */

#include <algorithm>
#include <boost/ref.hpp>

#include <mgpu/backend/transfer.hpp>
#include <mgpu/backend/dev_stream.hpp>
#include <mgpu/seg_dev_stream.hpp>
#include <mgpu/container/range_traits.hpp>
#include <mgpu/container/iterator_traits.hpp>

namespace mgpu
{

template <typename InputRange, typename OutputIterator, typename StreamType>
OutputIterator copy(const InputRange & input, OutputIterator output,
  StreamType const & stream);

namespace detail
{

/// device segmented to device segmented
template <typename InputRange, typename OutputIterator>
OutputIterator copy(const InputRange & input, OutputIterator output,
  is_segmented_tag, is_segmented_tag,
  device_memory_tag, device_memory_tag,
  seg_dev_stream const & stream = default_seg_stream)
{
  // start a copy for each segment
  std::size_t segments =
    ::mgpu::range_traits<InputRange>::segments(input);
  std::size_t offset = 0;
  for(std::size_t segment=0; segment<segments; segment++)
  {
    dev_rank_t rank =
      ::mgpu::range_traits<const InputRange>::rank(input, segment);
    invoke(
      ::mgpu::copy<
        typename ::mgpu::range_traits<const InputRange>::local_range,
        typename::mgpu::iterator_traits<OutputIterator>::local_iterator,
        backend::dev_stream
      >,
      make_range(input, segment),
      mgpu::iterator_traits<OutputIterator>::begin_local(output, segment),
      boost::ref(stream[rank]),
      rank
    );
    offset += ::mgpu::range_traits<const InputRange>::size(input, segment);
  }
  return output;
}

/// device segmented to device (gather from device to device)
template <typename InputRange, typename OutputIterator>
OutputIterator copy(const InputRange & input, OutputIterator output,
  is_segmented_tag, is_not_segmented_tag,
  device_memory_tag, device_memory_tag,
  seg_dev_stream const & stream = default_seg_stream)
{
  // start a copy for each segment
  std::size_t segments =
    ::mgpu::range_traits<InputRange>::segments(input);
  std::size_t offset = 0;
  for(std::size_t segment=0; segment<segments; segment++)
  {
    dev_rank_t rank =
      ::mgpu::range_traits<const InputRange>::rank(input, segment);
    invoke(
      ::mgpu::copy<
        typename ::mgpu::range_traits<const InputRange>::local_range,
        OutputIterator,
        backend::dev_stream
      >,
      make_range(input, segment),
      output + offset,
      boost::ref(stream[rank]),
      rank
    );
    offset += ::mgpu::range_traits<const InputRange>::size(input, segment);
  }
  return output;
}

/// device segmented to host (gather from device to host)
template <typename InputRange, typename OutputIterator>
OutputIterator copy(const InputRange & input, OutputIterator output,
  is_segmented_tag, is_not_segmented_tag,
  device_memory_tag, host_memory_tag,
  seg_dev_stream const & stream = default_seg_stream)
{
  // start a copy for each segment
  std::size_t segments =
    ::mgpu::range_traits<InputRange>::segments(input);
  std::size_t offset = 0;
  for(std::size_t segment=0; segment<segments; segment++)
  {
    dev_rank_t rank =
      ::mgpu::range_traits<const InputRange>::rank(input, segment);
    invoke(
      ::mgpu::copy<
        typename ::mgpu::range_traits<const InputRange>::local_range,
        OutputIterator,
        backend::dev_stream
      >,
      make_range(input, segment),
      output + offset,
      boost::ref(stream[rank]),
      rank
    );
    offset += ::mgpu::range_traits<const InputRange>::size(input, segment);
  }
  return output;
}

/// device to device segmented (scatter from device to device)
template <typename InputRange, typename OutputIterator>
OutputIterator copy(const InputRange & input, OutputIterator output,
  is_not_segmented_tag, is_segmented_tag,
  device_memory_tag, device_memory_tag,
  seg_dev_stream const & stream = default_seg_stream)
{
  // start a copy for each segment
  std::size_t segments =
    ::mgpu::iterator_traits<OutputIterator>::segments(output);

  for(std::size_t segment=0; segment<segments; segment++)
  {
    dev_rank_t rank =
      mgpu::iterator_traits<OutputIterator>::rank(output, segment);
    std::size_t segment_size =
      ::mgpu::iterator_traits<OutputIterator>::segment_size(output, segment);
    invoke(
      ::mgpu::copy<
           typename ::mgpu::range_traits<const InputRange>::make_range_type,
           typename::mgpu::iterator_traits<OutputIterator>::local_iterator,
           backend::dev_stream
          >,
      make_range(
        ::mgpu::range_traits<const InputRange>::begin(input) +
          (segment*segment_size),
        ::mgpu::range_traits<const InputRange>::begin(input) +
           ((segment+1)*segment_size)),
      mgpu::iterator_traits<OutputIterator>::begin_local(output, segment),
      boost::ref(stream[rank]),
      rank
    );
  }
  return output;
}

/// device to device (copy on device)
template <typename InputRange, typename OutputIterator>
OutputIterator copy(const InputRange & input, OutputIterator output,
  is_not_segmented_tag, is_not_segmented_tag,
  device_memory_tag, device_memory_tag,
  backend::dev_stream const & stream = backend::default_stream)
{
  backend::copy(
    ::mgpu::range_traits<const InputRange>::get_pointer(input),
    ::mgpu::iterator_traits<OutputIterator>::get_pointer(output),
    ::mgpu::range_traits<const InputRange>::size(input), stream
     );
  return output;
}

/// device to host (copy from device)
template <typename InputRange, typename OutputIterator>
OutputIterator copy(const InputRange & input, OutputIterator output,
  is_not_segmented_tag, is_not_segmented_tag,
  device_memory_tag, host_memory_tag,
  backend::dev_stream const & stream = backend::default_stream)
{
  backend::copy(
    ::mgpu::range_traits<const InputRange>::get_pointer(input),
    ::mgpu::iterator_traits<OutputIterator>::get_pointer(output),
    ::mgpu::range_traits<const InputRange>::size(input), stream
     );
  return output;
}

/// host to device segmented (scatter from host to device)
template <typename InputRange, typename OutputIterator>
OutputIterator copy(const InputRange & input, OutputIterator output,
  is_not_segmented_tag, is_segmented_tag,
  host_memory_tag, device_memory_tag,
  seg_dev_stream const & stream = default_seg_stream)
{
  // start a copy for each segment
  std::size_t segments =
    ::mgpu::iterator_traits<OutputIterator>::segments(output);

  for(std::size_t segment=0; segment<segments; segment++)
  {
    dev_rank_t rank =
      mgpu::iterator_traits<OutputIterator>::rank(output, segment);
    std::size_t segment_size =
      ::mgpu::iterator_traits<OutputIterator>::segment_size(output, segment);
    invoke(
      ::mgpu::copy<
           typename ::mgpu::range_traits<const InputRange>::make_range_type,
           typename::mgpu::iterator_traits<OutputIterator>::local_iterator,
           backend::dev_stream
          >,
      make_range(
        ::mgpu::range_traits<const InputRange>::begin(input) +
          (segment*segment_size),
        ::mgpu::range_traits<const InputRange>::begin(input) +
           ((segment+1)*segment_size)),
      mgpu::iterator_traits<OutputIterator>::begin_local(output, segment),
      boost::ref(stream[rank]),
      rank
    );
  }
  return output;
}

/// host to device (copy to device)
template <typename InputRange, typename OutputIterator>
OutputIterator copy(const InputRange & input, OutputIterator output,
  is_not_segmented_tag, is_not_segmented_tag,
  host_memory_tag, device_memory_tag,
  backend::dev_stream const & stream = backend::default_stream)
{
  backend::copy(
    ::mgpu::range_traits<const InputRange>::get_pointer(input),
    ::mgpu::iterator_traits<OutputIterator>::get_pointer(output),
    ::mgpu::range_traits<const InputRange>::size(input), stream
     );
  return output;
}

/// host to host (copy on host)
template <typename InputRange, typename OutputIterator>
OutputIterator copy(const InputRange & input, OutputIterator output,
  is_not_segmented_tag, is_not_segmented_tag,
  host_memory_tag, host_memory_tag,
  backend::dev_stream const & stream = backend::default_stream)
{
  std::copy(input.begin(), input.end(), output);
  return output;
}

} // namespace detail

} // namespace mgpu


#endif // MGPU_TRANSFER_DETAIL_COPY_HPP
