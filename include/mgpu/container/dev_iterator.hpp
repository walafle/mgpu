// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_DEV_ITERATOR_HPP
#define MGPU_CONTAINER_DEV_ITERATOR_HPP

/**
 * @file dev_iterator.hpp
 *
 * This header provides a device iterator
 */

#include <boost/iterator/iterator_facade.hpp>
#include <mgpu/core/dev_ptr.hpp>
#include <mgpu/core/ref.hpp>
#include <mgpu/backend/backend.hpp>

namespace mgpu
{

/**
 * @brief STL compatible iterator over device memory
 */
template <class Value>
class dev_iterator
  : public boost::iterator_facade< dev_iterator<Value>
                                 , Value
                                 , boost::random_access_traversal_tag
                                 >
{
  public:
    typedef Value                             value_type;

    typedef dev_ptr<Value>                    pointer;
    typedef dev_ptr<Value> const              const_pointer;

    typedef Value *                           raw_pointer;
    typedef Value * const                     const_raw_pointer;

    typedef std::size_t                       size_type;

  private:
    typedef ref<dev_ptr<Value> >              resource_type;

  public:

    /// construct empty device iterator
    dev_iterator() :
      resource_(0), offset_(0) {}

    /// construct device iterator from reference to device pointer
    explicit dev_iterator(ref<dev_ptr<Value> > p) :
      resource_(p), offset_(0) {}

    /// construct device iterator from reference to device pointer and offset
    explicit dev_iterator(ref<dev_ptr<Value> > p,
      const std::size_t & offset) : resource_(p), offset_(offset) {}

    /// return device id
    dev_id_t dev_id() const { return resource_.get().dev_id(); }

    /// return rank id
    dev_id_t get_rank_id() const { return resource_.get().get_rank_id(); }

    /// return offset
    std::size_t offset() const { return offset_; }

    /// access reference of device pointer
    ref<pointer> get_pointer_ref() { return resource_; }

    /// access reference of device pointer
    ref<const_pointer> get_pointer_ref() const
    { return ref<const_pointer>(resource_.get_pointer()); }

    /// access pointer
    pointer get_pointer() { return resource_.get() + offset_; }

    /// access constant pointer
    const_pointer get_pointer() const { return resource_.get() + offset_; }

    /// access raw pointer
    raw_pointer get_raw_pointer()
    { return resource_.get().get_raw_pointer() + offset_; }

    /// access constant raw pointer
    const_raw_pointer get_raw_pointer() const
    { return resource_.get().get_raw_pointer() + offset_; }

 private:

    friend class boost::iterator_core_access;

    Value& dereference() const
    {
      // dereferencing the pointer not possible since this is device memory
      assert(false);
      // avoid nvcc compiler warning
      return Value();
    }

    bool equal(const dev_iterator<Value> & other) const
    {
      return (resource_.get() == other.resource_.get() &&
        offset_ == other.offset_);
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

    std::ptrdiff_t distance_to(const dev_iterator<Value> & other) const
    {
      return other.offset_ - offset_;
    }


    /// reference to device pointer
    resource_type resource_;

    /// offset that must be applied to device pointer
    std::size_t offset_;
};


} // namespace mgpu

#endif // MGPU_CONTAINER_DEV_ITERATOR_HPP
