// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_DETAILS_DEV_SEG_VECTOR_ASYNC_IPP
#define MGPU_CONTAINER_DETAILS_DEV_SEG_VECTOR_ASYNC_IPP

/**
 * @file seg_dev_vector_async.ipp
 *
 * This header contains the segmented device vector asynchronous functions
 */

namespace mgpu
{

namespace detail
{


// asynchronous free methods
// _____________________________________________________________________________

template<typename T, typename Alloc>
void alloc_single_dev_vector_impl(
  typename seg_dev_vector<T, Alloc>::pointer * ptr,
  typename seg_dev_vector<T, Alloc>::size_type size)
{
  *ptr = backend::dev_malloc<T>(size);
}

template<typename T, typename Alloc>
void free_single_dev_vector_impl(
  typename seg_dev_vector<T, Alloc>::pointer * ptr)
{
  // free memory
  if(ptr != NULL)
  {
    backend::dev_free<T>(*ptr);
    delete ptr;
  }
}

} // namespace detail

} // namespace mgpu



#endif // MGPU_CONTAINER_DETAILS_DEV_SEG_VECTOR_ASYNC_IPP
