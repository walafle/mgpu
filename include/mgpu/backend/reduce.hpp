// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_REDUCE_HPP
#define MGPU_BACKEND_REDUCE_HPP

/**
 * @file reduce.hpp
 *
 * This header includes all the backend specific reduce code.
 */

#include <mgpu/config.hpp>

#include <mgpu/backend/cuda/reduce/reduce_operators.hpp>
#include <mgpu/backend/cuda/reduce/reduce_block.hpp>

namespace mgpu
{

namespace backend = backend_detail::MGPU_BACKEND;

} // namespace mgpu


#endif // MGPU_BACKEND_REDUCE_HPP

