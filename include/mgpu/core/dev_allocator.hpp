// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_DEV_ALLOCATOR_HPP
#define MGPU_CORE_DEV_ALLOCATOR_HPP

/*
 * Copyright 2008-2010 NVIDIA Corporation
 * Modified by Sebastian Schaetz
 */

/**
 * @file dev_allocator.hpp
 * 
 * This header provides a standard C++ allocator class for allocating device
 * memory.
 */

#include <stdexcept>
#include <limits>

#include <mgpu/core/dev_ptr.hpp>
#include <mgpu/backend/backend.hpp>
#include <mgpu/core/dev_set_scoped.hpp>

namespace mgpu
{

template<typename T> class dev_allocator;

template<>
class dev_allocator<void>
{
  public:
    typedef void                    value_type;
    typedef dev_ptr<void>           pointer;
    typedef const dev_ptr<void>     const_pointer;
    typedef std::size_t             size_type;
    typedef std::ptrdiff_t          difference_type;

    /**
     * @brief convert a dev_allocator<void> to dev_allocator<U>
     */
    template<typename U>
      struct rebind
    {
      typedef dev_allocator<U> other;
    };
};

/**
 * @brief dev_allocator is a standard memory allocator that utilizes
 * plattform-specific methods to allocate device memory
 * 
 * @tparam T type that the allocator handles
 *
 * @see http://www.sgi.com/tech/stl/Allocators.html
 * 
 * @ingroup core
 */
template<typename T>
class dev_allocator
{
  public:
    typedef T                       value_type;
    typedef dev_ptr<T>              pointer;
    typedef const dev_ptr<T>        const_pointer;
    typedef T&                      reference;
    typedef const T&                const_reference;
    typedef std::size_t             size_type;
    typedef std::ptrdiff_t          difference_type;

    /// convert a dev_allocator<T> to dev_allocator<U>
    template<typename U> struct rebind
    {
      typedef dev_allocator<U> other;
    };

    /// dev_allocator's null constructor does nothing
    dev_allocator() {}

    /// dev_allocator's null destructor does nothing
    ~dev_allocator() {}

    /// dev_allocator's copy constructor does nothing
    dev_allocator(dev_allocator const &) {}

    ///dev_allocator's template copy constructor does nothing
    template<typename U>
    dev_allocator(dev_allocator<U> const &) {}

    /// return the address of a reference
    pointer address(reference r) { return &r; }

    /// return the address of a const reference
    const_pointer address(const_reference r) { return &r; }

    /**
     * @brief allocate storage for objects on device
     *
     * @param size number of objects to allocate
     * 
     * @return pointer to the newly allocated objects
     */
    pointer allocate(size_type size, const_pointer = dev_ptr<T>())
    {
      if(size > this->max_size())
      {
        throw std::bad_alloc();
      }
      pointer result = backend::dev_malloc<value_type>(size);
      return result;
    }

    /**
     * @brief free storage for objects that were allocated on device
     *
     * @param p pointer to the previously allocated memory
     * 
     * @param count number of objects previously allocated at p
     */
    void deallocate(pointer p, size_type count)
    {
      // we need to make sure we call free in the correct device context
      dev_set_scoped d(p.dev_id());
      backend::dev_free(p);
    } 

    /**
     * @brief return how many elements can be allocated
     *
     * @return how many elements can be allocated
     */
    size_type max_size() const
    {
      return (std::numeric_limits<size_type>::max)() / sizeof(T);
    }

    /**
     * @brief tests if two dev_allocators are equal
     * 
     * @param x the other dev_allocator of interest
     * 
     * @return always returns true
     */
    bool operator==(dev_allocator const& x) { return true; }

    /**
     * @brief tests if two dev_allocator are not equal
     * 
     * @param x the other dev_allocator of interest
     * 
     * @return always returns false
     */
    bool operator!=(dev_allocator const &x)
    { 
      return !operator==(x); 
    }
    
    /// object construction does nothing
    void construct(pointer p, const T& val) { }

    /// object destruction does nothing
    void destroy(pointer p) { }
};


} // namespace mgpu




#endif // MGPU_CORE_DEV_ALLOCATOR_HPP
