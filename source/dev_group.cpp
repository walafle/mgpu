// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#include <mgpu/core/dev_group.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

namespace mgpu
{
  const dev_group all_devices = create_all_dev_group();
}
