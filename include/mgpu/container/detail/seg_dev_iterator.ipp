// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_DETAILS_SEG_DEV_ITERATOR_IPP
#define MGPU_CONTAINER_DETAILS_SEG_DEV_ITERATOR_IPP

/**
 * @file seg_dev_iterator.ipp
 *
 * This header contains the seg_dev_iterator class implementation
 */


namespace mgpu
{

template<typename Value, typename Alloc>
typename seg_dev_iterator<Value, Alloc>::pointer const 
  seg_dev_iterator<Value, Alloc>::get_pointer() const
{
  if(offset_)
  {
    std::size_t o = offset_;
    for(std::size_t segment=0; segment<resource_.segments(); segment++)
    {
      std::size_t s = resource_.size(segment);
      if(s >= o)
      {
        return resource_.get_pointer(segment) + o;
      }
      else
      {
        o -= s;
      }
    }
    throw std::out_of_range("seg_dev_iterator::raw() out of bounds");
  }
  return resource_.get_pointer(0);
}


template<typename Value, typename Alloc>
typename seg_dev_iterator<Value, Alloc>::pointer
  seg_dev_iterator<Value, Alloc>::get_pointer()
{
  // weird work-around to avoid code dublication
  const seg_dev_iterator<Value, Alloc> & cthis = *this;
  typename seg_dev_iterator<Value, Alloc>::pointer pt = cthis.get_pointer();
  return * const_cast<typename seg_dev_iterator<Value, Alloc>::pointer *>(&pt);
}

} // namespace mgpu



#endif // MGPU_CONTAINER_DETAILS_SEG_DEV_ITERATOR_IPP
