// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#include <vector>
#include <mgpu/container/dev_vector.hpp>
#include <mgpu/backend.hpp>
#include <mgpu/transfer/copy.hpp>

using namespace mgpu;

int main(void)
{
  // allocate memory
  std::vector<float> host_in(3, 42.), host_out(3, 0.);
  dev_vector<float> dev(3);

  // copy to device
  copy(host_in, dev.begin());

  // if more than 1 device: host -> device0 -> device1 -> host
  int devices = backend::get_dev_count();
  if(devices > 1)
  {
    dev_set_scoped s(1);
    dev_vector<float> dev2(3);
    // copy from device0 to device1 and back to host
    copy(dev, dev2.begin());
    copy(dev2, host_out.begin());
  }
  else // if not: host -> device -> host
  {
    copy(dev, host_out.begin());
  }

  printf("in %f %f %f | out %f %f %f\n", host_in[0], host_in[1],
    host_in[2], host_out[0], host_out[1], host_out[2]);
}
