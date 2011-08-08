// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_DEV_RANGE_HPP
#define MGPU_CONTAINER_DEV_RANGE_HPP

/**
 * @file dev_range.hpp
 */


namespace mgpu
{

template <class Value>
class dev_range
{
  public:
    typedef dev_iterator<Value>               iterator;
    typedef const dev_iterator<Value>         const_iterator;
    typedef Value                             value_type;

    typedef dev_ptr<Value>                    pointer;
    typedef const dev_ptr<Value>              const_pointer;

    typedef Value *                           raw_pointer;
    typedef Value * const                     const_raw_pointer;

    typedef std::size_t                       size_type;

  public:
    dev_range(const iterator & it, const size_type & size) :
      it_(it), size_(size)
    { }

    dev_range(const iterator & it1, const iterator & it2) :
      it_(it1), size_(it2-it1)
    { }

    /// return iterator to beginning of device range
    iterator begin() { return it_; }

    /// return constant iterator to beginning of device range
    const_iterator begin() const { return it_; }

    /// return iterator to end of device range
    iterator end()
    { return iterator(it_.get_pointer_ref(), it_.offset() + size_); }

    /// return constant iterator to end of device range
    const_iterator end() const
    { return iterator(it_.get_pointer_ref(), it_.offset() + size_); }

    //// returns the number of elements in this range
    size_type size() const { return size_; }

    /// access pointer
    pointer get_pointer() { return it_.get_pointer(); }

    /// access constant pointer
    const_pointer get_pointer() const { return it_.get_pointer(); }

    /// access raw pointer
    raw_pointer get_raw_pointer() { return it_.get_raw_pointer(); }

    /// access constant raw pointer
    const_raw_pointer get_raw_pointer() const { return it_.get_raw_pointer(); }

  private:

    /// iterator, beginning of range
    iterator it_;

    /// size of range
    size_type size_;

};


} // namespace mgpu

#endif // MGPU_CONTAINER_DEV_RANGE_HPP
