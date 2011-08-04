// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_UNIT_HELPER_FUNCTIONS_HPP
#define MGPU_UNIT_HELPER_FUNCTIONS_HPP

/**
 * @file helper_functions.hpp
 *
 * This header provides some helper functions
 */

#include <complex>


namespace mgpu
{

namespace unit
{

/// create a number from a value
template <typename T>
T make_number(const T & val)
{
  return val;
}

/// create a complex number from a value
template <typename T>
std::complex<T> make_number(T & val)
{
  return std::complex<T>(val, val);
}

} // namespace unit

} // namespace mgpu


#endif // MGPU_UNIT_HELPER_FUNCTIONS_HPP
