// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_DETAILS_SEG_DEV_VECTOR_IPP
#define MGPU_CONTAINER_DETAILS_SEG_DEV_VECTOR_IPP

/**
 * @file dev_dist_vector.ipp
 *
 * This header contains the seg_dev_vector class implementation
 */

#include <mgpu/container/detail/seg_dev_vector_async.ipp>
#include <mgpu/core/detail/splitter.hpp>
#include <mgpu/exception.hpp>
#include <mgpu/core/ref.hpp>
#include <mgpu/invoke.hpp>
#include <mgpu/synchronization.hpp>

namespace mgpu
{


// construct empty vector -----

template<typename T, typename Alloc>
seg_dev_vector<T, Alloc>::seg_dev_vector() : base(0, 0, 0, 0)
{
  reset_arrays();
}


// create segmented vector of size on the specified devices -----

template<typename T, typename Alloc>
seg_dev_vector<T, Alloc>::seg_dev_vector(size_type size,
  const rank_group & ranks) : base(size, 1, ranks.size(), size, ranks)
{
  allocate();
}

template<typename T, typename Alloc>
seg_dev_vector<T, Alloc>::seg_dev_vector(clone_size segmentsize,
  const rank_group & ranks) : base(segmentsize.get_size()*ranks.size(), 1,
    ranks.size(), segmentsize.get_size()*ranks.size(), ranks)
{
  allocate();
}


// construct a vector of size and distribute it across all ranks
// considering the blocksize

template<typename T, typename Alloc>
seg_dev_vector<T, Alloc>::seg_dev_vector(size_type size, size_type blocksize,
  const rank_group & ranks) :
  base(size, blocksize, ranks.size(), size/blocksize, ranks)

{
  allocate();
}


// named constructor to split a vector of fixed size across specified devices

template<typename T, typename Alloc>
seg_dev_vector<T, Alloc> seg_dev_vector<T, Alloc>::split(size_type size,
  const rank_group & ranks)
{
  return seg_dev_vector<T, Alloc>(size, ranks);
}


// named constructor to split a vector of fixed size across specified devices
// taking the blocksize under consideration

template<typename T, typename Alloc>
seg_dev_vector<T, Alloc> seg_dev_vector<T, Alloc>::split(size_type size,
  size_type blocksize, const rank_group & ranks)
{
  return seg_dev_vector<T, Alloc>(size, blocksize, ranks);
}


// named constructor to allocate a segmented vector across specified devices
// with each segment of size segmentsize

template<typename T, typename Alloc>
seg_dev_vector<T, Alloc> seg_dev_vector<T, Alloc>::clone(size_type segmentsize,
  const rank_group & ranks)
{
  return seg_dev_vector<T, Alloc>(clone_size(segmentsize), ranks);
}


// destroy vector -----

template<typename T, typename Alloc>
seg_dev_vector<T, Alloc>::~seg_dev_vector()
{
  for(std::size_t segment=0; segment<segments_; segment++)
  {
    invoke(detail::free_single_dev_vector_impl<T, Alloc>,
      storage_[segment].get_pointer(), ranks_[segment]);
  }
}


// set_null vector -----

template<typename T, typename Alloc>
void seg_dev_vector<T, Alloc>::set_null()
{
  for(std::size_t segment=0; segment<segments_; segment++)
  {
    invoke(&dev_vector<T, Alloc>::set_null, storage_[segment], ranks_[segment]);
  }
}


// reset member arrays of vector -----

template<typename T, typename Alloc>
void seg_dev_vector<T, Alloc>::reset_arrays()
{
  for(std::size_t segment=0; segment<MGPU_MAX_DEVICES; segment++)
  {
    sizes_[segment] = 0;
    blocks_[segment] = 0;
    storage_[segment].reset();
  }
}


// reset all members that are not vectors -----

template<typename T, typename Alloc>
void seg_dev_vector<T, Alloc>::reset_scalar_members()
{
  size_ = 0;
  blocksize_ = 0;
  segments_ = 0;
  overall_blocks_ = 0;
}


// allocate helper function -----
template<typename T, typename Alloc>
void seg_dev_vector<T, Alloc>::allocate()
{
  // initialize arrays with 0
  reset_arrays();

  // in some cases we have more devices than segments
  int segment_reduction = 0;

  // splitter might throw
  try
  {
    detail::splitter s(size_, blocksize_, segments_);

    // allocate vector
    for(unsigned int segment=0; segment<segments_; segment++)
    {
      size_type sz = s++;
      if(sz == 0)
      {
        segment_reduction++;
        continue;
      }
      sizes_[segment] = sz;
      blocks_[segment] = sz / blocksize_;

      // construct a vector with deferred allocation
      storage_[segment].set(new pointer());

      // trigger allocation
      invoke(detail::alloc_single_dev_vector_impl<T, Alloc>,
        storage_[segment].get_pointer(), sz, ranks_[segment]);
    }
  }
  catch(...)
  {
    reset_scalar_members();
    throw;
  }
  segments_ -= segment_reduction;
}

} // namespace mgpu



#endif // MGPU_CONTAINER_DETAILS_SEG_DEV_VECTOR_IPP
