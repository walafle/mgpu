// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_MAKE_RANGE_HPP
#define MGPU_CONTAINER_MAKE_RANGE_HPP

/**
 * @file make_range.hpp
 *
 * This header provides various methods of creating ranges
 */

#include <utility>
#include <vector>

#include <mgpu/container/dev_vector.hpp>
#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/container/dev_range.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace mgpu
{

/**
 * @brief create a range object from two iterators
 */
template <typename Iterator>
std::pair< Iterator, Iterator>
make_range(const Iterator & begin, const Iterator & end)
{
  return std::pair<Iterator, Iterator>(begin, end);
}

template <typename T>
dev_range<T>
make_range(const dev_iterator<T> & begin, const dev_iterator<T> & end)
{
  return dev_range<T>(begin, end-begin);
}

/**
 * @brief create a range object from a range (pass through)
 */
template <typename Iterator>
std::pair<Iterator, Iterator> &
make_range(std::pair< Iterator, Iterator> & range)
{
  return range;
}

template <typename Iterator>
const std::pair<Iterator, Iterator> &
make_range(const std::pair<Iterator, Iterator> & range)
{
  return range;
}

/**
 * @brief create a range object from a segmented device vector
 */
template <typename T, typename Alloc>
dev_range<T> make_range(seg_dev_vector<T, Alloc> & vec, const int & segment = 0)
{
  return vec.local(segment);
}

/**
 * @brief create a range object from a segmented vector
 */
template <typename T, typename Alloc>
const dev_range<T> make_range(const seg_dev_vector<T, Alloc> & vec,
  const int & segment = 0)
{
  return vec.local(segment);
}

/**
 * @brief create a range object from a device vector
 */
template <typename T, typename Alloc>
dev_range<T> make_range(dev_vector<T, Alloc> & vec, const int & segment = 0)
{
  return dev_range<T>(vec.begin(), vec.size());
}

/**
 * @brief create a range object from a device vector
 */
template <typename T, typename Alloc>
const dev_range<T> make_range(const dev_vector<T, Alloc> & vec,
  const int & segment = 0)
{
  return dev_range<T>(vec.begin(), vec.size());
}


/**
 * @brief create a range object from a std::vector
 */
template <typename T, typename Alloc>
std::pair< typename std::vector<T, Alloc>::iterator,
           typename std::vector<T, Alloc>::iterator>
make_range(std::vector<T, Alloc> & vec, const int & segment = 0)
{
  return std::pair< typename std::vector<T, Alloc>::iterator,
                    typename std::vector<T, Alloc>::iterator>
    (vec.begin(), vec.end());
}

/**
 * @brief create a range object from a const std::vector
 */
template <typename T, typename Alloc>
std::pair< typename std::vector<T, Alloc>::const_iterator,
           typename std::vector<T, Alloc>::const_iterator>
make_range(const std::vector<T, Alloc> & vec, const int & segment = 0)
{
  return std::pair< typename std::vector<T, Alloc>::const_iterator,
                    typename std::vector<T, Alloc>::const_iterator>
    (vec.begin(), vec.end());
}



/**
 * @brief create a range object from a boost ublas vector
 */
template <typename T, typename Alloc>
std::pair< typename boost::numeric::ublas::vector<T, Alloc>::iterator,
           typename boost::numeric::ublas::vector<T, Alloc>::iterator>
make_range(boost::numeric::ublas::vector<T, Alloc> & vec,
  const int & segment = 0)
{
  return std::pair< typename boost::numeric::ublas::vector<T, Alloc>::iterator,
                    typename boost::numeric::ublas::vector<T, Alloc>::iterator>
    (vec.begin(), vec.end());
}

/**
 * @brief create a range object from a const boost ublas vector
 */
template <typename T, typename Alloc>
std::pair< typename boost::numeric::ublas::vector<T, Alloc>::const_iterator,
           typename boost::numeric::ublas::vector<T, Alloc>::const_iterator>
make_range(const boost::numeric::ublas::vector<T, Alloc> & vec,
  const int & segment = 0)
{
  return std::pair< typename
                      boost::numeric::ublas::vector<T, Alloc>::const_iterator,
                    typename
                      boost::numeric::ublas::vector<T, Alloc>::const_iterator>
    (vec.begin(), vec.end());
}




} // namespace mgpu

#endif // MGPU_CONTAINER_MAKE_RANGE_HPP
