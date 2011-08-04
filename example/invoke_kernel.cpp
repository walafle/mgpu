// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#include <mgpu/invoke_kernel.hpp>
#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/synchronization.hpp>
#include <mgpu/environment.hpp>

using namespace mgpu;

template <typename T>
void kernel_caller(dev_range<T> dev,
  std::vector<std::size_t> & vec, dev_rank_t rank, bool select)
{
  // here you would call a kernel using backend syntax (e.g. CUDA)
  vec[rank] = dev.size();
  if(select) printf("rank %d was selected\n", rank);
}

int main(void)
{
  environment e;
  {
    seg_dev_vector<float> dev_vec(42);
    std::vector<std::size_t> sizes(e.size());
    invoke_kernel_all(kernel_caller<float>, dev_vec, sizes,
      pass_dev_rank, select_one);
    synchronize_barrier();
    for(unsigned int i=0; i<e.size(); i++)
      printf("rank %d size %lu\n", i, sizes[i]);
  }
}

