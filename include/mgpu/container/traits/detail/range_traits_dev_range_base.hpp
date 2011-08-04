// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_TRAITS_RANGE_TRAITS_DETAIL_RANGE_TRAITS_DEV_RANGE_BASE_HPP
#define MGPU_CONTAINER_TRAITS_RANGE_TRAITS_DETAIL_RANGE_TRAITS_DEV_RANGE_BASE_HPP

/**
 * @file range_traits_dev_range_base.hpp
 *
 * This header provides the range traits specialization that are shared by
 * device ranges (non segmented)
 */

namespace mgpu
{
namespace detail
{

/// define functions that are shared by device ranges (non segmented)
template <typename Range, typename PointerType, typename IteratorType>
struct range_traits_dev_range_base
{
  typedef typename Range::size_type size_type;

  static inline const size_type size(Range & range)
  { return range.size(); }

  static inline const size_type segments(Range & range)
  { return 1; }

  static inline const size_type increment(Range & range)
  { return 1; }


  static inline IteratorType begin(Range & range, const int & segment = 0)
  { return range.begin(); }

  static inline IteratorType end(Range & range, const int & segment = 0)
  { return range.end(); }

  static inline PointerType get_pointer(Range & range, const int & segment = 0)
  { return range.get_pointer(); }
};

} // namespace detail

} // namespace mgpu

#endif // MGPU_CONTAINER_TRAITS_RANGE_TRAITS_DETAIL_RANGE_TRAITS_DEV_RANGE_BASE_HPP
