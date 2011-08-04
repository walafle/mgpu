// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_2D_HPP
#define MGPU_CORE_2D_HPP

/** 
 * @file 2d.hpp
 *
 * This header provides helper structures for 2d algorithms
 */

namespace mgpu
{

/**
 * @brief structure describes an offset in a 2D  data structure
 * 
 * @ingroup main
 */
struct offset_2d
{
  unsigned int x;
  unsigned int y;
  
  /**
   * @brief create a 2D offset object
   */
  inline offset_2d(unsigned int x_offset, unsigned int y_offset) :
    x(x_offset), y(y_offset) {}

  /**
   * @brief create an empty 2D offset object
   */
  inline offset_2d() : x(0), y(0) {}
};

/**
 * @brief structure describes the dimensions of a 2D data structure
 * 
 * @ingroup main
 */
struct dim_2d
{
  unsigned int x;
  unsigned int y;
  
  static const unsigned int dims = 2;
  
  /**
   * @brief create a 2D dimension object
   */
  inline dim_2d(unsigned int x_dim, unsigned int y_dim) :
    x(x_dim), y(y_dim) {}
  
  /**
   * @brief create an empty 2D dimension object
   */
  inline dim_2d() : x(0), y(0) {}
  
  /**
   * @brief return dimension
   */
  inline unsigned int dim(unsigned short int i)
  {
    if(i==0) return x;
    if(i==1) return y;
    return 0;
  }
};

} // namespace mgpu


#endif // MGPU_CORE_2D_HPP
