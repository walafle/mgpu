// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_TRAITS_RANGE_TRAITS_POINTER_PAIR_HPP
#define MGPU_CONTAINER_TRAITS_RANGE_TRAITS_POINTER_PAIR_HPP

/**
 * @file range_traits_pointer_pair.hpp
 *
 * This header provides the range traits specialization for the pointer pair
 */

#include <mgpu/container/traits/detail/range_traits_pair_base.hpp>

namespace mgpu
{

template <typename T>
struct range_traits< std::pair< T*, T* > > :
  public detail::range_traits_pair_base<std::pair< T*, T* >, T*, T*>
{
  typedef std::pair< T*, T* > type;
  typedef const type const_type;
  typedef type make_range_type;

  // regular typedefs
  typedef T                                    value_type;
  typedef T &                                  reference;
  typedef T *                                  pointer;
  typedef T *                                  iterator;
  typedef T * const                            const_iterator;
  typedef std::size_t                          size_type;

  // mgpu specific typedefs
  typedef host_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;
};

template <typename T>
struct range_traits<const std::pair< T*, T* > > :
  public detail::range_traits_pair_base<const std::pair< T*, T* >,
   T const *,  T const * >
{
  typedef const std::pair< T*, T* > type;
  typedef type const_type;
  typedef type make_range_type;

  // regular typedefs
  typedef T                                    value_type;
  typedef T &                                  reference;
  typedef T * const                            pointer;
  typedef T * const                            iterator;
  typedef T * const                            const_iterator;
  typedef std::size_t                          size_type;

  // mgpu specific typedefs
  typedef host_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;

};




} // namespace mgpu

#endif // MGPU_CONTAINER_TRAITS_RANGE_TRAITS_POINTER_PAIR_HPP
