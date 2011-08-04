// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_TRAITS_RANGE_TRAITS_ITERATOR_PAIR_HPP
#define MGPU_CONTAINER_TRAITS_RANGE_TRAITS_ITERATOR_PAIR_HPP

/**
 * @file range_traits_iterator_pair.hpp
 *
 * This header provides the range traits specialization for the iterator pair
 */

#include <mgpu/container/traits/detail/range_traits_pair_base.hpp>

namespace mgpu
{


template <typename Iterator>
struct range_traits< std::pair< Iterator, Iterator > > :
  public detail::range_traits_pair_base<std::pair< Iterator, Iterator >,
    typename Iterator::value_type *, Iterator>
{
  typedef std::pair< Iterator, Iterator > type;
  typedef const type const_type;
  typedef type make_range_type;

  // regular typedefs
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::reference         reference;
  typedef value_type *                         pointer;
  typedef Iterator                             iterator;
  typedef const Iterator                       const_iterator;
  typedef std::size_t                          size_type;

  // mgpu specific typedefs
  typedef host_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;

};

template <typename Iterator>
struct range_traits<const std::pair< Iterator, Iterator > >:
  public detail::range_traits_pair_base<const std::pair< Iterator, Iterator >,
    typename Iterator::value_type const * , Iterator const>
{
  typedef const std::pair< Iterator, Iterator > type;
  typedef type const_type;
  typedef type make_range_type;

  // regular typedefs
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::reference         reference;
  typedef const value_type *                   pointer;
  typedef const Iterator                       iterator;
  typedef const Iterator                       const_iterator;
  typedef std::size_t                          size_type;

  // mgpu specific typedefs
  typedef host_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;

};




} // namespace mgpu

#endif // MGPU_CONTAINER_TRAITS_RANGE_TRAITS_ITERATOR_PAIR_HPP
