// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_RANGE_TRAITS_HPP
#define MGPU_CONTAINER_RANGE_TRAITS_HPP

/**
 * @file range_traits.hpp
 *
 * This header provides the range traits
 */

#include <vector>

#include <boost/assert.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <mgpu/container/dev_vector.hpp>
#include <mgpu/container/dev_range.hpp>
#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/container/tags.hpp>

namespace mgpu
{

// generic versions that assert -----

template <typename Range>
struct range_traits
{
  typedef Range type;
  typedef const type const_type;
  typedef Range make_range_type;

  // regular typedefs
  typedef typename Range::value_type        value_type;
  typedef typename Range::reference         reference;
  typedef typename Range::pointer           pointer;
  typedef typename Range::iterator          iterator;
  typedef typename Range::const_iterator    const_iterator;
  typedef typename Range::size_type         size_type;

  // mgpu specific typedefs
  typedef no_memory_tag location_tag;
  typedef no_segmentation_information_tag segmented_tag;

  // mgpu specific functions
  static inline const size_type size(type &)
  {
    BOOST_ASSERT_MSG(false, "range_traits::size() invalid range");
    return size_type();
  }

  static inline const size_type segments(type &)
  {
    BOOST_ASSERT_MSG(false, "range_traits::segments() invalid range");
    return size_type();
  }

  static inline const size_type increment(type &)
  {
    BOOST_ASSERT_MSG(false, "range_traits::increment() invalid range");
    return size_type();
  }


  static inline iterator begin(type &, const int & segment = 0)
  {
    BOOST_ASSERT_MSG(false, "range_traits::begin() invalid range");
    return iterator();
  }

  static inline iterator end(type &, const int & segment = 0)
  {
    BOOST_ASSERT_MSG(false, "range_traits::end() invalid range");
    return iterator();
  }


  static inline pointer get_pointer(type & it, const int & segment = 0)
  {
    BOOST_ASSERT_MSG(false, "range_traits::end() invalid range");
    return pointer();
  }
};

} // namespace mgpu

#include <mgpu/container/traits/range_traits_vector.hpp>
#include <mgpu/container/traits/range_traits_iterator_pair.hpp>
#include <mgpu/container/traits/range_traits_pointer_pair.hpp>
#include <mgpu/container/traits/range_traits_boost_ublas_vector.hpp>
#include <mgpu/container/traits/range_traits_dev_vector.hpp>
#include <mgpu/container/traits/range_traits_dev_range.hpp>
#include <mgpu/container/traits/range_traits_seg_dev_vector.hpp>

#endif // MGPU_CONTAINER_RANGE_TRAITS_HPP
