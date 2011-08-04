// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_TRAITS_RANGE_TRAITS_DEV_VECTOR_HPP
#define MGPU_CONTAINER_TRAITS_RANGE_TRAITS_DEV_VECTOR_HPP

/**
 * @file range_traits_dev_vector.hpp
 *
 * This header provides the range traits specialization for the device vector
 */

#include <mgpu/container/traits/detail/range_traits_dev_range_base.hpp>

namespace mgpu
{

template <typename T>
struct range_traits<dev_vector<T> > :
  public detail::range_traits_dev_range_base<dev_vector<T>,
    typename dev_vector<T>::pointer,
    typename dev_vector<T>::iterator>
{
  typedef dev_vector<T> type;
  typedef const type const_type;
  typedef dev_range<T> make_range_type;

  // regular typedefs
  typedef typename type::value_type        value_type;
  typedef typename type::pointer           pointer;
  typedef typename type::iterator          iterator;
  typedef typename type::const_iterator    const_iterator;
  typedef typename type::size_type         size_type;

  // mgpu specific typedefs
  typedef device_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;


};

template <typename T>
struct range_traits<const dev_vector<T> > :
  public detail::range_traits_dev_range_base<const dev_vector<T>,
    typename dev_vector<T>::const_pointer,
    typename dev_vector<T>::const_iterator>
{
  typedef const dev_vector<T> type;
  typedef type const_type;
  typedef const dev_range<T> make_range_type;

  // regular typedefs
  typedef typename type::value_type        value_type;
  typedef typename type::const_pointer     pointer;
  typedef typename type::const_iterator    iterator;
  typedef typename type::const_iterator    const_iterator;
  typedef typename type::size_type         size_type;

  // mgpu specific typedefs
  typedef device_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;

};


} // namespace mgpu

#endif // MGPU_CONTAINER_TRAITS_RANGE_TRAITS_DEV_VECTOR_HPP
