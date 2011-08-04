// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_DEV_GROUP_HPP
#define MGPU_CORE_DEV_GROUP_HPP

/**
 * @file dev_group.hpp
 *
 * This header provides the device group class
 */


#include <mgpu/config.hpp>
#include <mgpu/core/dev_id.hpp>
#include <mgpu/core/detail/group_base.hpp>
#include <mgpu/backend/dev_management.hpp>

namespace mgpu
{

typedef detail::group_base<dev_id_t, MGPU_MAX_DEVICES> dev_group;

inline dev_group create_dev_group(const dev_id_t & from, const dev_id_t & to)
{
  return detail::group_base<dev_id_t, MGPU_MAX_DEVICES>::from_to(from, to);
}

inline dev_group create_all_dev_group()
{
  int to = backend::get_dev_count();
  return detail::group_base<dev_id_t, MGPU_MAX_DEVICES>::from_to(0, to);
}

extern const dev_group all_devices;

} // namespace mgpu


#endif // MGPU_CORE_DEV_GROUP_HPP
