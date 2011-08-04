// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----

// tool to detect the number of GPUs in the systems

#include <cuda_runtime.h>

int main(void)
{
  int num;
  cudaError_t r = cudaGetDeviceCount(&num);
  if(r == cudaSuccess)
  {
    return num;
  }
  else if(r == cudaErrorNoDevice)
  {
    return 0;
  }
  return -1;
}

