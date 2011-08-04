// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_SEG_DEV_ITERATOR_HPP
#define MGPU_CONTAINER_SEG_DEV_ITERATOR_HPP

/**
 * @file seg_dev_iterator.hpp
 *
 * This header provides a device segmented iterator
 */

#include <stdexcept>

#include <boost/iterator/iterator_facade.hpp>
#include <mgpu/core/dev_ptr.hpp>
#include <mgpu/core/ref.hpp>
#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/backend/backend.hpp>

namespace mgpu
{

template<typename T, typename Alloc>
class seg_dev_vector;

/**
 * @brief STL compatible iterator over device memory
 */
template <typename Value, typename Alloc = dev_allocator<Value> >
class seg_dev_iterator
  : public boost::iterator_facade< seg_dev_iterator<Value, Alloc>
                                 , Value
                                 , boost::random_access_traversal_tag
                                 >
{
  public:
    typedef Value                                 value_type;
    typedef Value const                           const_value_type;

    typedef dev_ptr<Value>                        pointer;
    typedef dev_ptr<Value> const                  const_pointer;

    typedef dev_iterator<Value>                   local_iterator;
    typedef dev_iterator<Value> const             const_local_iterator;

    typedef typename Alloc::size_type             size_type;

  private:
    typedef seg_dev_vector<Value, Alloc> &  resource_type;

  public:

    /// construct with reference to seg_dev_vector and an offset
    seg_dev_iterator(resource_type resource, const size_type & offset = 0) :
      resource_(resource), offset_(offset) {}

    /// return constant device pointer interpreting offset as a global offset
    const_pointer get_pointer() const;

    /// return device pointer interpreting offset as a global offset
    pointer get_pointer();

    /// return device pointer of segment n and
    /// interpreting offset as a local offset
    pointer get_pointer(const int & segment)
    { return resource_.get_pointer(segment) + offset_; }

    /// return constant device pointer of segment n and
    /// interpreting offset as a local offset
    const_pointer get_pointer(const int & segment) const
    { return resource_.get_pointer(segment) + offset_; }

    /// return number of segments
    size_type segments() const { return resource_.segments(); };

    /// return iterator to first element of segment n of container
    local_iterator begin_local(const int & segment)
    { return resource_.begin_local(segment) + offset_; }

    /// return constant iterator to first element of segment n of container
    const_local_iterator begin_local(const int & segment) const
    { return resource_.begin_local(segment) + offset_; }

    /// return the rank of the device on which the segment resides
    dev_rank_t rank(const int & segment) const
    { return resource_.rank(segment); }

    /// return the size of the container
    inline size_type size() const { return resource_.size(); }

    /// return the size of segment n
    inline size_type size(const int & segment) const
    { return resource_.size(segment); }

  private:
    friend class boost::iterator_core_access;

    Value& dereference() const
    {
      // dereferencing the pointer not possible since this is device memory
      assert(false);
      // avoid nvcc compiler warning
      return Value();
    }

    bool equal(const seg_dev_iterator<Value> & other) const
    {
      return (&resource_ == &other.resource_ && offset_ == other.offset_);
    }

    void increment()
    {
      ++offset_;
    }

    void decrement()
    {
      --offset_;
    }

    void advance(std::ptrdiff_t n)
    {
      offset_ += n;
    }

    std::ptrdiff_t distance_to(const seg_dev_iterator<Value> & other) const
    {
      return other.offset_ - offset_;
    }

    /// constant reference to segmented device vector
    resource_type resource_;

    /// offset accumulated iterator modifications
    size_type offset_;

};

} // namespace mgpu

#include <mgpu/container/detail/seg_dev_iterator.ipp>

#endif // MGPU_CONTAINER_SEG_DEV_ITERATOR_HPP
