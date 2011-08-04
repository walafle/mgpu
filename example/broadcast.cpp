// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#include <vector>
#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/transfer/copy.hpp>
#include <mgpu/transfer/broadcast.hpp>
#include <mgpu/synchronization.hpp>

using namespace mgpu;

int main(void)
{
  environment e;
  {
    // allocate memory
    std::vector<float> host_in(1, 42.), host_out(e.size(), 0.);
    seg_dev_vector<float> dev(e.size());

    // copy to device and from device
    broadcast(host_in, dev.begin());
    copy(dev, host_out.begin());
    synchronize_barrier();
    for(unsigned int i=0; i<e.size(); i++)
      printf("in: %f out %f\n", host_in[0], host_out[i]);
  }
}
