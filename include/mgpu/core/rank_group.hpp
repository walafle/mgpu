// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_RANK_GROUP_HPP
#define MGPU_CORE_RANK_GROUP_HPP

/**
 * @file rank_group.hpp
 *
 * This header provides the rank group class
 */


#include <mgpu/config.hpp>
#include <mgpu/core/dev_rank.hpp>
#include <mgpu/core/detail/group_base.hpp>

namespace mgpu
{

typedef detail::group_base<dev_rank_t, MGPU_MAX_DEVICES> rank_group;

} // namespace mgpu


#endif // MGPU_CORE_RANK_GROUP_HPP
