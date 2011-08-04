// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_TRAITS_RANGE_TRAITS_DETAIL_RANGE_TRAITS_PAIR_BASE_HPP
#define MGPU_CONTAINER_TRAITS_RANGE_TRAITS_DETAIL_RANGE_TRAITS_PAIR_BASE_HPP

/**
 * @file range_traits_pair_base.hpp
 *
 * This header provides the range traits specialization that are shared by
 * ranges composed from pairs
 */

namespace mgpu
{
namespace detail
{

/// define functions that are shared by device ranges (non segmented)
template <typename Range, typename PointerType, typename IteratorType>
struct range_traits_pair_base
{
  typedef std::size_t size_type;

  static inline size_type size(Range & range)
  { return range.second - range.first; }

  static inline size_type segments(Range & range)
  { return 1; }

  static inline size_type increment(Range & range)
  { return 1; }


  static inline IteratorType begin(Range & range, const int & segment = 0)
  { return range.first; }

  static inline IteratorType end(Range & range, const int & segment = 0)
  { return range.second; }


  static inline PointerType get_pointer(Range & range, const int & segment = 0)
  { return &(*range.first); }
};

} // namespace detail

} // namespace mgpu

#endif // MGPU_CONTAINER_TRAITS_RANGE_TRAITS_DETAIL_RANGE_TRAITS_PAIR_BASE_HPP
