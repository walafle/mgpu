// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_UNIT_SEQUENCES_HPP
#define MGPU_UNIT_SEQUENCES_HPP

/**
 * @file sequences.hpp
 *
 * This header provides sequences that can be used with e.g. std::generate
 */

#include <complex>


namespace mgpu
{

namespace unit
{

/// rising sequence generator that can be used with e.g. std::generate
template <typename T>
struct rising_sequence
{
  T val;
  rising_sequence() : val(0) {};
  T operator()() { return val = val+1; }
};

template <typename T>
struct rising_sequence<std::complex<T> >
{
  std::complex<T> val;
  rising_sequence() : val(0, 0) {};
  std::complex<T> operator()() { return val = val+std::complex<T>(.1, .1); }
};


/// random sequence generator that can be used with e.g. std::generate
template <typename T>
struct random_sequence
{
  T operator()() { return rand()%10; }
};

template <typename T>
struct random_sequence<std::complex<T> >
{
  std::complex<T> operator()() { return std::complex<T>(rand()%10, rand()%10); }
};


} // namespace unit

} // namespace mgpu


#endif // MGPU_UNIT_TEST_TYPES_HPP
