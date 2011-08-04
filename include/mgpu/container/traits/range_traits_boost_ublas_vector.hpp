// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_TRAITS_RANGE_TRAITS_BOOST_UBLAS_VECTOR_HPP
#define MGPU_CONTAINER_TRAITS_RANGE_TRAITS_BOOST_UBLAS_VECTOR_HPP

/**
 * @file range_traits_boost_ublas_vector.hpp
 *
 * This header provides the range traits specialization for the boost ublas
 * vector
 */

#include <mgpu/container/traits/detail/range_traits_range_base.hpp>

namespace mgpu
{

template <typename T, typename Alloc>
struct range_traits<boost::numeric::ublas::vector<T, Alloc> > :
  public detail::range_traits_range_base<boost::numeric::ublas::vector<T, Alloc>,
    typename boost::numeric::ublas::vector<T, Alloc>::pointer,
    typename boost::numeric::ublas::vector<T, Alloc>::iterator>
{
  typedef boost::numeric::ublas::vector<T, Alloc> type;
  typedef const boost::numeric::ublas::vector<T, Alloc> const_type;
  typedef std::pair<typename type::iterator, typename type::iterator>
    make_range_type;

  // regular typedefs
  typedef typename type::value_type        value_type;
  typedef typename type::reference         reference;
  typedef typename type::pointer           pointer;
  typedef typename type::iterator          iterator;
  typedef typename type::const_iterator    const_iterator;
  typedef std::size_t                      size_type;

  // mgpu specific typedefs
  typedef host_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;

};


template <typename T, typename Alloc>
struct range_traits<const boost::numeric::ublas::vector<T, Alloc> >:
  public detail::range_traits_range_base<const boost::numeric::ublas::vector<T, Alloc>,
    typename boost::numeric::ublas::vector<T, Alloc>::const_pointer,
    typename boost::numeric::ublas::vector<T, Alloc>::const_iterator>
{
  typedef const boost::numeric::ublas::vector<T, Alloc> type;
  typedef type const_type;
  typedef std::pair<typename type::const_iterator,
                    typename type::const_iterator>
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

#endif // MGPU_CONTAINER_TRAITS_RANGE_TRAITS_BOOST_UBLAS_VECTOR_HPP
