// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_DEV_PTR_PACK_HPP
#define MGPU_CORE_DEV_PTR_PACK_HPP

/**
 * @file dev_ptr_pack.hpp
 *
 * This header provides the device pointer pack class
 */

#include <boost/array.hpp>
#include <mgpu/config.hpp>
#include <mgpu/core/dev_ptr.hpp>

namespace mgpu
{

/**
 * @brief device pointer pack
 *
 * A collection of device pointers 
 *
 * @ingroup core
 */

template <typename T>
class dev_ptr_pack
{
  public:

    /**
     * @brief empty device pointer pack constructor
     */
    inline dev_ptr_pack() : size_(0), pack_() { }

    /**
     * @brief set the size of the pack
     */
    inline void set_size_(const std::size_t size) { size_ = size; }

    /// operator []
    dev_ptr<T>& operator [](const int & offset) { return pack_[offset]; }

    /**
     * @brief return the size of the pack
     */
    inline std::size_t size() const { return size_; }

    /**
     * @brief post increment operator
     */
    dev_ptr_pack<T> & operator ++()
    { 
      for(int i=0; i<MGPU_MAX_DEVICES; i++)
      {
        ++pack_[i];
      }
      return *this; 
    }
 
    /**
     * @brief post increment operator
     */
    dev_ptr_pack<T> & operator --()
    { 
      for(int i=0; i<MGPU_MAX_DEVICES; i++)
      {
        --pack_[i];
      }
      return *this; 
    } 

    /**
     * @brief increment the device pointer and return it
     *
     * @param b value by which the pointer should be incremented
     *
     * @return incremented device pointer
     */
    dev_ptr_pack<T>& operator +=(const std::size_t & b)
    {
      for(int i=0; i<MGPU_MAX_DEVICES; i++)
      {
        pack_+=b;
      }
      return *this;
    }

  private:

    /// number of device pointers in the pack
    std::size_t size_;

    /// array containing the device pionter pack 
    boost::array<dev_ptr<T>, MGPU_MAX_DEVICES> pack_;
};


} // namespace mgpu


#endif // MGPU_CORE_DEV_PTR_PACK_HPP

