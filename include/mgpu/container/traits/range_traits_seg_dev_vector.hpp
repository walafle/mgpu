// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_TRAITS_RANGE_TRAITS_SEG_DEV_VECTOR_HPP
#define MGPU_CONTAINER_TRAITS_RANGE_TRAITS_SEG_DEV_VECTOR_HPP

/**
 * @file range_traits_seg_dev_vector.hpp
 *
 * This header provides the range traits specialization for the segmented device
 * vector
 */

namespace mgpu
{


namespace detail
{

/// define functions that are shared by all seg_dev_vector range traits
template <typename Range, typename PointerType, typename LocalIteratorType>
struct range_traits_seg_dev_vector
{
  typedef typename Range::size_type size_type;

  static inline const size_type size(Range & range, const int segment = -1)
  { return (segment==-1) ? range.size() : range.size(segment); }

  static inline const size_type segments(const Range & range)
  { return range.segments(); }

  static inline dev_rank_t rank(Range & range, const int & segment)
  { return range.rank(segment); }

  static inline const size_type increment(Range & range)
  { return 1; }


  static inline LocalIteratorType begin(Range & range, const int & segment = 0)
  { return range.begin_local(segment); }

  static inline LocalIteratorType end(Range & range, const int & segment = 0)
  { return range.end_local(segment); }


  static inline PointerType get_pointer(Range & range, const int & segment = 0)
  { return range.get_pointer(segment); }
};

} // namespace detail


template <typename T, typename Alloc>
struct range_traits<seg_dev_vector<T, Alloc> > :
  public detail::range_traits_seg_dev_vector<
    seg_dev_vector<T, Alloc>,
    typename seg_dev_vector<T, Alloc>::pointer,
    typename seg_dev_vector<T, Alloc>::local_iterator>
{
  typedef seg_dev_vector<T, Alloc> type;
  typedef const type const_type;
  typedef dev_range<T> make_range_type;

  // regular typedefs
  typedef typename type::value_type           value_type;
  typedef typename type::pointer              pointer;
  typedef typename type::local_iterator       iterator;
  typedef typename type::const_local_iterator const_iterator;
  typedef typename type::size_type            size_type;
  typedef typename type::local_range          local_range;
  typedef typename type::const_local_range    const_local_range;

  // mgpu specific typedefs
  typedef device_memory_tag location_tag;
  typedef is_segmented_tag segmented_tag;

};

template <typename T, typename Alloc>
struct range_traits<const seg_dev_vector<T, Alloc> > :
  public detail::range_traits_seg_dev_vector<
    const seg_dev_vector<T, Alloc>,
    typename seg_dev_vector<T, Alloc>::const_pointer,
    typename seg_dev_vector<T, Alloc>::const_local_iterator>
{
  typedef const seg_dev_vector<T, Alloc> type;
  typedef type const_type;
  typedef dev_range<T> make_range_type;

  // regular typedefs
  typedef typename type::value_type             value_type;
  typedef typename type::const_pointer          pointer;
  typedef typename type::const_local_iterator   iterator;
  typedef typename type::const_local_iterator   const_iterator;
  typedef typename type::size_type              size_type;
  typedef typename type::const_local_range      local_range;
  typedef typename type::const_local_range      const_local_range;

  // mgpu specific typedefs
  typedef device_memory_tag location_tag;
  typedef is_segmented_tag segmented_tag;

};

} // namespace mgpu

#endif // MGPU_CONTAINER_TRAITS_RANGE_TRAITS_SEG_DEV_VECTOR_HPP
