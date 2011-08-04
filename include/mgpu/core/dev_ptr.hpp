// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_DEV_PTR_HPP
#define MGPU_CORE_DEV_PTR_HPP

/** 
 * @file dev_ptr.hpp
 *
 * This header provides a generic device pointer class
 */

#include <stddef.h>

#include <mgpu/core/dev_id.hpp>
#include <mgpu/backend/backend.hpp>
#include <mgpu/exception.hpp>
#include <mgpu/environment.hpp>

namespace mgpu
{

/**
 * @brief a generic device pointer class that stores type information
 * 
 * @ingroup core
 */
template <typename T>
struct dev_ptr
{
  /// pointer type
  typedef T * pointer;

  /// const pointer type
  typedef const T * const_pointer;

  /// value type the device pointer holds
  typedef T value_type;

  public:

    /**
     * @brief create pointer that points nowhere
     */
    dev_ptr() : ptr_(NULL), dev_(no_device) {}

    /**
     * @brief create device pointer that points to ptr
     *
     * @param ptr pointer of type T that identifies device memory
     */
    dev_ptr(T * const & ptr) : ptr_(ptr), dev_(backend::dev_id(ptr)) {}

    /**
     * @brief create device pointer that points to ptr
     *
     * @param ptr pointer of type T that identifies device memory
     *
     * @param device on which device is this pointer allocated
     */
    dev_ptr(T * const & ptr, const dev_id_t & device) :
      ptr_(ptr), dev_(device) {}
    


    /**
     * @biref access the raw pointer
     *
     * @return the raw pointer of type T that identifies device memory
     */
    pointer get_raw_pointer() { return ptr_; }

    /**
     * @biref access the raw pointer
     *
     * @return the raw pointer of type T that identifies device memory
     */
    const_pointer get_raw_pointer() const { return ptr_; }

    /**
     * @brief access the device id
     *
     * @return the id of the device the pointer points to
     */
    dev_id_t dev_id() const { return dev_; }

    /**
     * @brief access device rank
     *
     * Can only be used if environment is initialized
     *
     * @return the rank of the device the pointer points to
     */
    dev_rank_t rank() const { return environment::rank(dev_); }

    /**
     * @brief set pointer to NULL
     */
    void set_null() { ptr_ = NULL; dev_ = no_device; }

    /**
     * @brief return if pointer is NULL
     */
    bool is_null()
    {
      return (ptr_ == NULL && dev_ == no_device);
    }

    /**
     * @brief assign other dev_ptr to this dev_ptr
     *
     * @param b dev_ptr that should be assigned
     *
     * @return reference to this dev_ptr
     */
    dev_ptr<T>& operator =(dev_ptr<T> const & b)
    {
      ptr_ = b.ptr_;
      dev_ = b.dev_;
      return *this;
    }

    /**
     * @brief increment the device pointer and return a new one
     *
     * @param b value by which the pointer should be incremented
     *
     * @return incremented device pointer
     */
    dev_ptr<T> operator +(const std::size_t & b) const
    {
      T * x = ptr_ + b;
      return dev_ptr<T>(x, dev_);
    }

    /**
     * @brief increment the device pointer and return it
     *
     * @param b value by which the pointer should be incremented
     *
     * @return incremented device pointer
     */
    dev_ptr<T>& operator +=(const std::size_t & b)
    {
      ptr_+=b;
      return *this;
    }

    /**
     * @brief increment the device pointer by one
     *
     * @return incremented device pointer
     */
    dev_ptr<T> & operator ++()
    {
      ++ptr_;
      return *this;
    }

    /**
     * @brief decrement the device pointer by one
     *
     * @return decrement device pointer
     */
    dev_ptr<T> & operator --()
    {
      --ptr_;
      return *this;
    }

    /**
     * @brief decrement the device pointer and return a new one
     *
     * @param b value by which the pointer should be decremented
     *
     * @return decrement device pointer
     */
    dev_ptr<T> operator -(const std::size_t & b) const
    {
      T * x = ptr_ - b;
      return dev_ptr<T>(x, dev_);
    }

    /**
     * @brief subtract one ptr from another ptr
     *
     * @param other pointer
     *
     * @return difference in elements between pointers
     */
    std::ptrdiff_t operator -(const dev_ptr<T> & other) const
    {
      if(dev_ != other.dev_)
      {
        MGPU_THROW("tried to subtract dev_ptrs on different devices");
      }
      return ptr_ - other.ptr_;
    }

    /**
     * @brief compare two device pointers
     *
     * @param b pointer that should be compared
     *
     * @return true if pointers are equal, else false
     */
    bool operator ==(const dev_ptr<T> & b) const
    {
      return (dev_ == b.dev_ && ptr_ == b.ptr_);
    }

    /**
     * @brief compare two device pointers
     *
     * @param b pointer that should be compared
     *
     * @return true if pointers are unequal, else false
     */
    bool operator !=(const dev_ptr<T>  & b) const
    {
      return (dev_ != b.dev_ || ptr_ != b.ptr_);
    }

  private:
  
    /// actual pointer that identifies device memory
    T * ptr_;

    /// id of device the pointer points to
    dev_id_t dev_;
};


} // namespace mgpu


#endif // MGPU_CORE_DEV_PTR_HPP
