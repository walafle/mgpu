#ifndef MGPU_CORE_HOST_OPT_ALLOCATOR_HPP
#define MGPU_CORE_HOST_OPT_ALLOCATOR_HPP

/*
 * Copyright 2008-2010 NVIDIA Corporation
 * Modified by Sebastian Schaetz
 */

/**
 * @file host_opt_allocator.hpp
 * 
 * This header provides a standard C++ allocator class for allocating host
 * memory in a plattform specific optimized manner
 */

#include <stdexcept>
#include <limits>

#include <mgpu/backend/host_allocation.hpp>

namespace mgpu
{


template<typename T> class host_opt_allocator;

template<>
  class host_opt_allocator<void>
{
  public:
    typedef void           value_type;
    typedef void       *   pointer;
    typedef const void *   const_pointer;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    /**
     * @brief convert a host_opt_allocator<void> to host_opt_allocator<U>
     */
    template<typename U>
      struct rebind
    {
      typedef host_opt_allocator<U> other;
    };
};

/**
 * @brief host_opt_allocator is a standard memory allocator that utilizes
 * plattform-specific optimized methods to allocate memory.
 * 
 * For the CUDA plattformt this allocates pinned memory which speeds up memory 
 * transfers between host and device since the memory is trackable by the CUDA
 * runtime which allows DMA.
 * 
 * @tparam T type that the allocator handles
 *
 * @see http://www.sgi.com/tech/stl/Allocators.html
 * 
 * @ingroup core
 */
template<typename T>
  class host_opt_allocator
{
  public:
    typedef T              value_type;
    typedef T*             pointer;
    typedef const T*       const_pointer;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    /// convert a host_opt_allocator<T> to host_opt_allocator<U>
    template<typename U> struct rebind
    {
      typedef host_opt_allocator<U> other;
    };

    /// host_opt_allocator's null constructor does nothing
    inline host_opt_allocator() {}

    /// host_opt_allocator's null destructor does nothing
    inline ~host_opt_allocator() {}

    /// host_opt_allocator's copy constructor does nothing
    inline host_opt_allocator(host_opt_allocator const &) {}

     /// templated host_opt_allocator's copy constructor does nothing
    template<typename U>
    inline host_opt_allocator(host_opt_allocator<U> const &) {}

    /// return the address of a reference
    inline pointer address(reference r) { return &r; }

    /// return the address of a const reference
    inline const_pointer address(const_reference r) { return &r; }

    /**
     * @brief allocate storage for objects in an optimized way
     *
     * @param cnt number of objects to allocate
     * 
     * @return pointer to the newly allocated objects
     */
    inline pointer allocate(size_type cnt, const_pointer = 0)
    {
      if(cnt > this->max_size())
      {
        throw std::bad_alloc();
      }
      pointer result = backend::host_opt_malloc<value_type>(cnt);
      return result;
    }

    /**
     * @brief free storage for objects that were allocated in an optimized way 
     *
     * @param p pointer to the previously allocated memory
     * 
     * @param cnt number of objects previously allocated at p
     */
    inline void deallocate(pointer p, size_type cnt)
    {
      backend::host_opt_free<value_type>(p);
    } 

    /**
     * @brief return how many elements can be allocated
     *
     * @return how many elements can be allocated
     */
    inline size_type max_size() const
    {
      return (std::numeric_limits<size_type>::max)() / sizeof(T);
    }

    /**
     * @brief tests if two host_opt_allocators are equal
     * 
     * @param x the other host_opt_allocator of interest
     * 
     * @return always returns true
     */
    inline bool operator==(host_opt_allocator const& x) { return true; }

    /**
     * @brief tests if two host_opt_allocator are not equal
     * 
     * @param x the other host_opt_allocator of interest
     * 
     * @return always returns false
     */
    inline bool operator!=(host_opt_allocator const &x) 
    { 
      return !operator==(x); 
    }
    
    /**
     * @brief construct an object
     * 
     * Constructs an object of type T (the template parameter) on the 
     * location pointed by p using its copy constructor to initialize its value 
     * to val.
     * 
     * @param p pointer to locatioin where T should be constructed
     * 
     * @param val value to initialize the constructed element to
     */
    inline void construct(pointer p, const T& val)
    {
      ::new((void *)p) T(val);
    }

    /**
     * @brief destroy an object
     * 
     * Destroys the object of type T pointed by p.
     * 
     * @param p pointer to object that should be destroyed
     */
    inline void destroy(pointer p)
    {
      p->~T();
    }
};


} // namespace mgpu




#endif // MGPU_CORE_HOST_OPT_ALLOCATOR_HPP
