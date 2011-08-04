// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_ITERATOR_TRAITS_HPP
#define MGPU_CONTAINER_ITERATOR_TRAITS_HPP

/**
 * @file iterator_traits.hpp
 *
 * This header contains the iterator traits
 */

#include <vector>

#include <boost/mpl/int.hpp>
#include <boost/assert.hpp>

#include <mgpu/container/dev_iterator.hpp>
#include <mgpu/container/seg_dev_iterator.hpp>
#include <mgpu/container/tags.hpp>

namespace mgpu
{

// the following genric versions are taken from boost/detail/iterator.hpp
// copyright David Abrahams 2002


// generic version -----

template <class Iterator>
struct iterator_traits
{
  typedef Iterator type;
  typedef const type const_type;

  // regular typedefs
  typedef type iterator;
  typedef type local_iterator;
  typedef typename Iterator::value_type value_type;
  typedef typename Iterator::reference reference;
  typedef typename Iterator::pointer pointer;
  typedef typename Iterator::difference_type difference_type;
  typedef typename Iterator::iterator_category iterator_category;

  // mgpu specific typedefs
  typedef host_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;

  // mgpu specific functions
  static inline std::size_t increment(const_type &) { return 1; }

  static inline pointer get_pointer(type & it, const int & segment = 0)
  { return &(*it); }

  static inline pointer get_raw_pointer(type & it, const int & segment = 0)
  { return &(*it); }
};


// pointer specialization -----

template <typename T>
struct iterator_traits<T*>
{
  typedef T* type;
  typedef const type const_type;

  // regular typedefs
  typedef type iterator;
  typedef type local_iterator;
  typedef T value_type;
  typedef T& reference;
  typedef T* pointer;
  typedef std::ptrdiff_t difference_type;
  typedef std::random_access_iterator_tag iterator_category;

  // mgpu specific typedefs
  typedef host_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;

  // mgpu specific functions
  static inline std::size_t increment(const_type &) { return 1; }

  static inline T* get_pointer(type & it, const int & segment = 0)
  { return it; }

  static inline T* get_raw_pointer(type & it, const int & segment = 0)
  { return it; }
};


// device pointer specialization -----

template <typename T>
struct iterator_traits<dev_ptr<T> >
{
  typedef dev_ptr<T> type;
  typedef const type const_type;

  typedef type                                      iterator;
  typedef const_type                                const_iterator;

  typedef type                                      local_iterator;
  typedef const_type                                const_local_iterator;

  typedef typename type::value_type                 value_type;

  typedef typename type::raw_pointer                raw_pointer;
  typedef typename type::const_raw_pointer          const_raw_pointer;

  typedef typename type::pointer                    pointer;
  typedef typename type::const_pointer              const_pointer;

  typedef typename type::size_type                  size_type;


  typedef device_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;

  // mgpu specific functions
  static inline std::size_t increment(const_type &) { return 1; }

  static inline pointer get_pointer(type & it, const int & segment = 0)
  { return it; }

  static inline pointer get_raw_pointer(type & it, const int & segment = 0)
  { return it.get_raw_pointer(); }
};

template <typename T>
struct iterator_traits<const dev_ptr<T> >
{
  typedef const dev_ptr<T> type;
  typedef type const_type;


  typedef type                                      iterator;
  typedef const_type                                const_iterator;

  typedef type                                      local_iterator;
  typedef const_type                                const_local_iterator;

  typedef typename type::value_type                 value_type;

  typedef typename type::const_raw_pointer          raw_pointer;
  typedef typename type::const_raw_pointer          const_raw_pointer;

  typedef typename type::const_pointer              pointer;
  typedef typename type::const_pointer              const_pointer;

  typedef typename type::size_type                  size_type;


  typedef typename iterator::difference_type difference_type;
  typedef std::random_access_iterator_tag iterator_category;

  // mgpu specific typedefs
  typedef device_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;

  // mgpu specific functions
  static inline std::size_t increment(const_type &) { return 1; }

  static inline pointer get_pointer(type & it, const int & segment = 0)
  { return it; }

  static inline pointer get_raw_pointer(type & it, const int & segment = 0)
  { return it.get_raw_pointer(); }
};


// device iterator specialization -----

template <typename T>
struct iterator_traits<dev_iterator<T> >
{
  typedef dev_iterator<T> type;
  typedef type const_type;

  typedef type                                      iterator;
  typedef const_type                                const_iterator;

  typedef type                                      local_iterator;
  typedef const_type                                const_local_iterator;

  typedef typename type::value_type                 value_type;

  typedef typename type::raw_pointer                raw_pointer;
  typedef typename type::const_raw_pointer          const_raw_pointer;

  typedef typename type::pointer                    pointer;
  typedef typename type::const_pointer              const_pointer;

  typedef typename type::size_type                  size_type;

  typedef typename iterator::difference_type difference_type;
  typedef typename iterator::iterator_category iterator_category;


  typedef device_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;


  // mgpu specific functions
  static inline std::size_t increment(const_type &) { return 1; }

  static inline pointer get_pointer(type & it, const int & segment = 0)
  { return it.get_pointer(); }

  static inline raw_pointer get_raw_pointer(type & it, const int & segment = 0)
  { return it.get_pointer().get_raw_pointer(); }
};

template <typename T>
struct iterator_traits<const dev_iterator<T> >
{
  typedef const dev_iterator<T> type;
  typedef type const_type;


  typedef type                                      iterator;
  typedef const_type                                const_iterator;

  typedef type                                      local_iterator;
  typedef const_type                                const_local_iterator;

  typedef typename type::value_type                 value_type;

  typedef typename type::const_raw_pointer          raw_pointer;
  typedef typename type::const_raw_pointer          const_raw_pointer;

  typedef typename type::const_pointer              pointer;
  typedef typename type::const_pointer              const_pointer;

  typedef typename type::size_type                  size_type;

  typedef typename iterator::difference_type difference_type;
  typedef typename iterator::iterator_category iterator_category;


  typedef device_memory_tag location_tag;
  typedef is_not_segmented_tag segmented_tag;


  // mgpu specific functions
  static inline std::size_t increment(const_type &) { return 1; }

  static inline pointer get_pointer(type & it, const int & segment = 0)
  { return it.get_pointer(); }

  static inline raw_pointer get_raw_pointer(type & it, const int & segment = 0)
  { return it.get_pointer().get_raw_pointer(); }
};


// segmented device iterator specialization -----

template <typename T, typename Alloc>
struct iterator_traits<seg_dev_iterator<T, Alloc> >
{
  typedef const seg_dev_iterator<T, Alloc> type;
  typedef type const_type;


  typedef typename type::value_type                 value_type;
  typedef typename type::const_value_type           conts_value_type;

  typedef typename type::local_iterator             local_iterator;
  typedef typename type::const_local_iterator       const_local_iterator;

  typedef typename type::pointer                    pointer;
  typedef typename type::const_pointer              const_pointer;

  typedef typename type::difference_type            difference_type;
  typedef typename type::iterator_category          iterator_category;

  typedef typename type::size_type                  size_type;


  typedef device_memory_tag location_tag;
  typedef is_segmented_tag segmented_tag;

  // mgpu specific functions
  static inline size_type const increment(const_type &) { return 1; }

  static inline pointer get_pointer(type & it, const int & segment = 0)
  { return it.get_pointer(segment); }

  static inline local_iterator begin_local(type & it, const int & segment)
  { return it.begin_local(segment); }

  static inline size_type segments(const_type & it) { return it.segments(); }

  static inline size_type segment_size(const_type & it, const int & segment)
  { return it.size(segment); }

  static inline dev_rank_t rank(type & it, const int & segment)
  { return it.rank(segment); }
};

template <typename T, typename Alloc>
struct iterator_traits<const seg_dev_iterator<T, Alloc> >
{
  typedef const seg_dev_iterator<T, Alloc> type;
  typedef type const_type;


  typedef typename type::const_value_type           value_type;
  typedef typename type::const_value_type           conts_value_type;

  typedef typename type::const_local_iterator       local_iterator;
  typedef typename type::const_local_iterator       const_local_iterator;

  typedef typename type::const_pointer              pointer;
  typedef typename type::const_pointer              const_pointer;

  typedef typename type::difference_type            difference_type;
  typedef typename type::iterator_category          iterator_category;

  typedef typename type::size_type                  size_type;


  typedef device_memory_tag location_tag;
  typedef is_segmented_tag segmented_tag;

  // mgpu specific functions
  static inline size_type const increment(const_type &) { return 1; }

  static inline pointer get_pointer(type & it, const int & segment = 0)
  { return it.get_pointer(segment); }

  static inline local_iterator begin_local(type & it, const int & segment)
  { return it.begin_local(segment); }

  static inline size_type segments(const_type & it) { return it.segments(); }

  static inline size_type segment_size(const_type & it, const int & segment)
  { return it.size(segment); }

  static inline dev_rank_t rank(type & it, const int & segment)
  { return it.rank(segment); }
};


} // namespace mgpu

#endif // MGPU_CONTAINER_ITERATOR_TRAITS_HPP
