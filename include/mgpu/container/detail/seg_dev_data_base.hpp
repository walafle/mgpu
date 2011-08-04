// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_DETAIL_SEG_DEV_DATA_BASE_HPP
#define MGPU_CONTAINER_DETAIL_SEG_DEV_DATA_BASE_HPP

/**
 * @file seg_dev_data_base.hpp
 *
 * Contains basic data and methods of segmented constructs
 */

namespace mgpu
{

namespace detail
{

template <typename T, typename Pointer>
class seg_dev_data_base
{
  public:

    typedef T                                             value_type;
    typedef const T                                       const_value_type;

    typedef T *                                           raw_pointer;
    typedef T const *                                     const_raw_pointer;

    typedef Pointer                                       pointer;
    typedef const Pointer                                 const_pointer;

    typedef dev_iterator<T>                               local_iterator;
    typedef const dev_iterator<T>                         const_local_iterator;

    typedef dev_range<T>                                  local_range;
    typedef const dev_range<T>                            const_local_range;

    typedef T &                                           raw_reference;
    typedef const T &                                     const_raw_reference;

    typedef std::size_t                                   size_type;

    typedef boost::array<size_type, MGPU_MAX_DEVICES>     sizes_type;
    typedef boost::array<ref<Pointer>, MGPU_MAX_DEVICES>  storage_type;

    typedef int const &                                   offset_type;

  public:

    /// construct with various sizes
    seg_dev_data_base(size_type size, size_type blocksize,
      size_type segments, size_type overall_blocks) :
      size_(size), blocksize_(blocksize),
      segments_(segments), overall_blocks_(overall_blocks)
      {}

    /// construct with various sizes and ranks
    seg_dev_data_base(size_type size, size_type blocksize,
      size_type segments, size_type overall_blocks, rank_group ranks) :
      size_(size), blocksize_(blocksize),
      segments_(segments), overall_blocks_(overall_blocks), ranks_(ranks)
      {}

    /// return the raw pointer of segment n
    raw_pointer get_raw_pointer(offset_type offset)
    { return storage_[offset].get().get_raw_pointer(); }

    /// return the raw pointer of segment n
    const_raw_pointer get_raw_pointer(offset_type offset) const
    { return storage_[offset].get().get_raw_pointer(); }

    /// return the device pointer of segment n
    pointer get_pointer(offset_type offset)
    { return storage_[offset].get(); }

    /// return the device pointer of segment n
    const_pointer get_pointer(offset_type offset) const
    { return storage_[offset].get(); }


    /// return a local range
    local_range local(const int & segment)
    { return local_range(this->begin_local(segment), sizes_[segment]); }

    /// return a local range
    const_local_range local(const int & segment) const
    { return local_range(this->begin_local(segment), sizes_[segment]); }


    /// return iterator to first element of segment n of container
    local_iterator begin_local(offset_type offset)
    { return local_iterator(storage_[offset]); }

    /// return iterator to first element of segment n of container
    const_local_iterator begin_local(offset_type offset) const
    { return local_iterator(storage_[offset]); }


    /// return iterator to one past last element of segment n of container
    local_iterator end_local(offset_type offset)
    { return local_iterator(storage_[offset], sizes_[offset]); }

    /// return iterator to one past last element of segment n of container
    const local_iterator end_local(offset_type offset) const
    { return local_iterator(storage_[offset], sizes_[offset]); }


    /// return the size of the container
    size_type size() const { return size_; }

    /// return the size of segment n
    size_type size(offset_type offset) const { return sizes_[offset]; }


    /// return the id of the device on which the segment resides
    dev_id_t dev_id(offset_type offset) const
    { return storage_[offset].get().dev_id(); }


    /// return the rank of the device on which the segment resides
    dev_rank_t rank(offset_type offset) const
    { return ranks_[offset]; }


    /// return the number of segments
    size_type segments() const { return segments_; }

    /// return the size of the blocks
    size_type blocksize() const { return blocksize_; }

    /// return the overall number of blocks
    size_type blocks() const { return overall_blocks_; }

    /// return the number of blocks in segment n
    size_type blocks(offset_type offset) const { return blocks_[offset]; }

  protected:

    /// pointer to the data
    storage_type storage_;


    /// number of elements in entire segmented vector
    size_type size_;

    /// blocksize
    size_type blocksize_;

    /// number of segments in this distributed vector
    size_type segments_;

    /// number of blocks in entire distributed vector (size_/blocksize_)
    size_type overall_blocks_;


    /// number of elements in each segment
    sizes_type sizes_;

    /// number of blocks in each device vector
    sizes_type blocks_;


    /// device group
    rank_group ranks_;
};

} // namespace detail

} // namespace mgpu

#include <mgpu/container/detail/seg_dev_iterator.ipp>

#endif // MGPU_CONTAINER_DETAIL_SEG_DEV_DATA_BASE_HPP
