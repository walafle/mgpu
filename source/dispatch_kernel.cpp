// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#include <mgpu/invoke_kernel.hpp>

namespace mgpu
{
  const detail::pass_dev_id_ pass_dev_id = detail::pass_dev_id_();
  const detail::pass_dev_rank_ pass_dev_rank = detail::pass_dev_rank_();
  const detail::select_one_ select_one = detail::select_one_();
}
