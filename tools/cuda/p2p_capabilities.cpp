// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----


#include <cuda_runtime.h>

#include <stdio.h>

int main(void)
{
  int num;
  cudaError_t r = cudaGetDeviceCount(&num);
  if(r != cudaSuccess)
  {
    return -1;
  }

  printf("{{");
  for(int from=0; from<num; from++)
  {
    for(int to=0; to<num; to++)
    {
      int can_access;
      cudaError_t r = cudaDeviceCanAccessPeer(&can_access, from, to);
      if(r != cudaSuccess) can_access = 0;

      if(can_access) printf("%d", 1);
      else printf("%d", 0);

      if(to < num-1) printf(",");
    }
    if(from != num-1) printf("},\n {");
    else printf("}}");
  }
  return 1;
}
