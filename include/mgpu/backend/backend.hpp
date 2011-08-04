// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_BACKEND_HPP
#define MGPU_BACKEND_BACKEND_HPP

/**
 * @file backend.hpp
 *
 * This header includes all the backend specific code.
 */

#include <mgpu/config.hpp>

#include <mgpu/backend/cuda/cuda_call.hpp>
#include <mgpu/backend/cuda/dev_allocation.hpp>
#include <mgpu/backend/cuda/dev_exception.hpp>
#include <mgpu/backend/dev_management.hpp>
#include <mgpu/backend/cuda/dev_stream.hpp>
#include <mgpu/backend/cuda/host_allocation.hpp>
#include <mgpu/backend/cuda/transfer.hpp>


namespace mgpu
{

namespace backend = backend_detail::MGPU_BACKEND;

} // namespace mgpu


#endif // MGPU_BACKEND_BACKEND_HPP

