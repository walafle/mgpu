// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_TAGS_HPP
#define MGPU_CONTAINER_TAGS_HPP

/**
 * @file tags.hpp
 *
 * This header provides the range traits
 */

#include <boost/mpl/int.hpp>

namespace mgpu
{

// location tags -----

struct no_memory_tag : public boost::mpl::int_<-1> {};

struct host_memory_tag : public boost::mpl::int_<0> {};

struct device_memory_tag : public boost::mpl::int_<1> {};


// segmented or not segmented tags -----

struct no_segmentation_information_tag : public boost::mpl::int_<-1> {};

struct is_not_segmented_tag : public boost::mpl::int_<0> {};

struct is_segmented_tag : public boost::mpl::int_<1> {};



} // namespace mgpu

#endif // MGPU_CONTAINER_TAGS_HPP
