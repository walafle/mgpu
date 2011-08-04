// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_DEV_VECTOR_HPP
#define MGPU_CONTAINER_DEV_VECTOR_HPP

/** 
 * @file dev_vector.hpp
 *
 * This header provides a generic device vector class
 */
 
#include <stddef.h>
#include <mgpu/core/dev_allocator.hpp>
#include <mgpu/container/dev_iterator.hpp>
#include <mgpu/container/dev_range.hpp>

namespace mgpu
{


/**
 * @brief device vector class
 * 
 * @tparam T type the vector holds
 * 
 * @tparam Alloc allocator
 * 
 * @note This class is written in a best effort manner and modeled after 
 * std::vector. Some things that are part of std::vector do not make sense for a 
 * dev_vector and are thus ommited. Also if functionality is not requried at the 
 * moment that would otherwise make sense it is also not implemented. The class 
 * is however modeled after std::vector.
 * 
 * @ingroup core
 */
template<typename T, typename Alloc = dev_allocator<T> >
class dev_vector
{
  private:
    typedef typename Alloc::pointer         storage_type;
    
  public:

    typedef typename Alloc::value_type      value_type;
    typedef typename Alloc::size_type       size_type;

    typedef T *                             raw_pointer;
    typedef const T *                       const_raw_pointer;

    typedef typename Alloc::pointer         pointer;
    typedef typename Alloc::const_pointer   const_pointer;

    typedef dev_iterator<T>                 iterator;
    typedef const dev_iterator<T>           const_iterator;

    typedef Alloc                           allocator_type;


    /**
     * @brief construct an empty vector
     */
    dev_vector() : storage_(), size_(0), allocator_() { }

    /**
     * @brief create a vector of size
     * 
     * @param size size of vector
     */
    explicit dev_vector(size_type size) : storage_(), size_(size), allocator_()
    { storage_ = allocator_.allocate(size_); }
    
    /**
     * @brief create a vector from existing pointer and size
     *
     * @param ptr pointer to memory
     *
     * @param size size of vector
     */
    explicit dev_vector(pointer ptr, size_type size) :
      storage_(ptr), size_(size), allocator_() { }

    /**
     * @brief create a vector of size but don't allocate
     *
     * @param size size of vector
     */
    explicit dev_vector(size_type size, bool deferred_allocation) :
      storage_(), size_(size), allocator_()
    {
      if(!deferred_allocation)
      {
        storage_ = allocator_.allocate(size_);
      }
    }

    /**
     * @brief allocate
     */
    void allocate()
    {
      if(storage_.is_null())
      {
        storage_ = allocator_.allocate(size_);
      }
    }

    /**
     * @brief destroy the vector
     */
    ~dev_vector()
    { clear(); }

    /**
     * @brief clear content of vector
     */
    void clear()
    {
      if(size_ > 0)
      {
        allocator_.deallocate(storage_, size_);
        size_ = 0;
      }
    }
        
    /**
     * @brief returns the number of elements in this dev_vector
     * 
     * @return number of elements in this vector
     */
    size_type size() const { return size_; }
    

    /// return iterator to beginning of device vector
    iterator begin()
    { return dev_iterator<value_type>(ref<storage_type>(storage_)); }

    /// return constant iterator to beginning of device vector
    const_iterator begin() const
    {
      // we can cast the const away here since we return a const_iterator
      return dev_iterator<value_type>(
        ref<storage_type>(*const_cast<storage_type *>(&storage_)));
    }

    /// return iterator to end of device vector
    iterator end()
    { return dev_iterator<value_type>(ref<storage_type>(storage_), size_); }

    /// return constant iterator to end of device vector
    const_iterator end() const
    {
      // see above for rational
      return dev_iterator<value_type>(
        ref<storage_type>(*const_cast<storage_type *>(&storage_)), size_);
    }


    /// return device pointer to first element of device vector
    pointer get_pointer() { return storage_; }

    /// return constant device  pointer to first element of device vector
    const_pointer get_pointer() const { return storage_; }


    /// return pointer to first element of device vector
    raw_pointer get_raw_pointer() { return storage_.get_raw_pointer(); }

    /// return constant pointer to first element of device vector
    const_raw_pointer get_raw_pointer() const
    { return storage_.get_raw_pointer(); }


    /// fill with set_null bytes
    void set_null() { backend::dev_set(storage_, 0, size_); };

  protected:
    /// storage
    storage_type storage_;

    /// size of vector with number of elements
    size_type size_; 
    
    /// allocator
    allocator_type allocator_;

};


} // namespace mgpu


#endif // MGPU_CONTAINER_DEV_VECTOR_HPP
