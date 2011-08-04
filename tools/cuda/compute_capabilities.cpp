// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----


// tool to detect the compute capabilities of the devices in the system

#include <set>
#include <utility>
#include <stdio.h>

#include <cuda_runtime.h>

int main(void)
{
  std::set<std::pair<int, int> > capabilities;
  typedef std::set<std::pair<int, int> >::iterator It;

  int num;
  cudaError_t r = cudaGetDeviceCount(&num);
  if(r == cudaSuccess)
  {
    for(int i=0; i<num; i++)
    {
      cudaDeviceProp properties;
      r = cudaGetDeviceProperties(&properties, i);
      if(r != cudaSuccess)
      {
        return -1;
      }
      capabilities.insert(
        std::pair<int, int>(properties.major, properties.minor));
    }
  }
  else if(r == cudaErrorNoDevice)
  {
    return 0;
  }
  bool first = true;
  for(It i = capabilities.begin(); i != capabilities.end(); i++)
  {
    if(first)
    {
      printf("%d%d", i->first, i->second);
    }
    else
    {
      printf(" %d%d ", i->first, i->second);
    }
    first = false;
  }
  return -1;
}
