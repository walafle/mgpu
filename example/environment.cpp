// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#include <mgpu/environment.hpp>
#include <mgpu/backend.hpp>

using namespace mgpu;

int main(void)
{
  {
    // all available devices
    environment e;
  }

  int devices = backend::get_dev_count();
  if(devices > 1)
  {
    // explicitly specify all available devices
    environment e(dev_group::from_to(0, devices));
  }

  if(devices > 2)
  {
    // use devices 0, 1 and 2
    environment e(dev_group(0, 1, 2));
  }
}
