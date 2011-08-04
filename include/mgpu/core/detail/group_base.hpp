// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_DETAIL_GROUP_BASE_HPP
#define MGPU_CORE_DETAIL_GROUP_BASE_HPP

/**
 * @file group_base.hpp
 *
 * This header provides a base class for various groups
 */

#include <boost/array.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <mgpu/exception.hpp>
#include <mgpu/config.hpp>

namespace mgpu
{
namespace detail
{

/**
 * @brief group_base
 *
 * A group base class
 *
 * @ingroup core
 */
template <typename T, std::size_t max_size>
class group_base
{
  public:

    /**
     * @brief create group that contains Ts from from to to-1
     *
     * @param from first T
     * @param to one past last T
     *
     * @return created group
     */
    static inline group_base<T, max_size> from_to(const T & from, const T & to)
    {
      group_base<T, max_size> g;
      for(int i=from, j=0; i<to; i++, j++)
      {
        g.group_[j] = i;
      }
      g.size_ = to-from;
      return g;
    }

    /**
     * @brief empty group constructor
     */
    inline group_base() : size_(0) { }

    /**
     * @brief construct group with T
     *
     * There are more constructors that take more Ts, with the maximum
     * being the maximum number of devices supported by the library
     */
    inline group_base(T i0) : size_(1) { group_[0] = i0; }


    #define MGPU_GROUP_BASE_ARGS(z, n, _) , T id ## n
    #define MGPU_GROUP_BASE_ASSIGN(z, n, _) group_[n] = id ## n;

    #define MGPU_GROUP_BASE_CTOR(z, n, _)                                      \
      inline group_base(T id0                                                  \
      BOOST_PP_REPEAT_FROM_TO(1, n, MGPU_GROUP_BASE_ARGS, _) ) :               \
        size_(n)                                                               \
      { BOOST_PP_REPEAT_FROM_TO(0, n, MGPU_GROUP_BASE_ASSIGN, _) }             \
      /**/

    // constructors with 0...MGPU_MAX_DEVICES
    BOOST_PP_REPEAT_FROM_TO(2,
      BOOST_PP_INC(MGPU_MAX_DEVICES), MGPU_GROUP_BASE_CTOR, _)

    /// operator []
    T & operator [](const int & offset) { return group_[offset]; }

    /// operator []
    const T operator [](const int & offset) const
    { return group_[offset]; }

    /// operator <<=
    group_base & operator <<=(const T& id)
    {
      if(size_ == max_size)
      {
        MGPU_THROW("device group full");
      }
      group_[size_] = id;
      size_++;
      return *this;
    }

    /**
     * @brief return the size of the group
     */
    inline const std::size_t & size() const { return size_; }

    /**
     * @brief set the size of the group
     */
    inline void set_size_(const std::size_t size) { size_ = size; }

  private:

    /// number of Ts in group
    std::size_t size_;

    /// array containing list of devices
    boost::array<T, max_size> group_;
};


} // namespace detail

} // namespace mgpu


#endif // MGPU_CORE_DETAIL_GROUP_BASE_HPP
