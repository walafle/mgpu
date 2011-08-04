// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#include <mgpu/config.hpp>

#ifdef MGPU_USING_CUDA_BACKEND

#include <mgpu/backend/cuda/dev_stream.hpp>

namespace mgpu
{
namespace backend_detail
{
namespace cuda
{

const dev_stream default_stream(0);

} // cuda

} // backend_detail

} // namespace mgpu

#endif // MGPU_USING_CUDA_BACKEND
