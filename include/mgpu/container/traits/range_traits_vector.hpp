// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_TRAITS_RANGE_TRAITS_VECTOR_HPP
#define MGPU_CONTAINER_TRAITS_RANGE_TRAITS_VECTOR_HPP

/**
 * @file range_traits_vector.hpp
 *
 * This header provides the range traits specialization for the std vector
 */

#include <mgpu/container/traits/detail/range_traits_range_base.hpp>

namespace mgpu
{

template <typename T, typename Alloc>
struct range_traits<std::vector<T, Alloc> > :
  public detail::range_traits_range_base<std::vector<T, Alloc>,
    typename std::vector<T, Alloc>::pointer,
    typename std::vector<T, Alloc>::iterator>
{
  typedef std::vector<T, Alloc> type;
  typedef const std::vector<T, Alloc> const_type;
  typedef std::pair< typename std::vector<T, Alloc>::iterator,
                     typename std::vector<T, Alloc>::iterator >
  make_range_type;

  // regular typedefs
  typedef typename type::value_type        value_type;
  typedef typename type::reference         reference;
  typedef typename type::pointer           pointer;
  typedef typename type::iterator          iterator;
  typedef typename type::const_iterator    const_iterator;
  typedef typename type::size_type         size_type;

  // mgpu specific typedefs
  typedef host_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;
};

template <typename T, typename Alloc>
struct range_traits<const std::vector<T, Alloc> > :
  public detail::range_traits_range_base<const std::vector<T, Alloc>,
    typename std::vector<T, Alloc>::const_pointer,
    typename std::vector<T, Alloc>::const_iterator>
{
    typedef const std::vector<T, Alloc> type;
    typedef std::vector<T, Alloc> const_type;
    typedef std::pair< typename std::vector<T, Alloc>::const_iterator,
                       typename std::vector<T, Alloc>::const_iterator >
    make_range_type;

    // regular typedefs
    typedef typename type::value_type        value_type;
    typedef typename type::reference         reference;
    typedef typename type::const_pointer     pointer;
    typedef typename type::const_iterator    iterator;
    typedef typename type::const_iterator    const_iterator;
    typedef typename type::size_type         size_type;

    // mgpu specific typedefs
    typedef host_memory_tag location_tag;
    typedef is_not_segmented_tag segmented_tag;
};

} // namespace mgpu

#endif // MGPU_CONTAINER_TRAITS_RANGE_TRAITS_VECTOR_HPP
