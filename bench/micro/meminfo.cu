// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



/**
 * Memory Info Test
 */

#include <stdio.h>
#include <mgpu/backend/dev_management.hpp>

using namespace mgpu;


int main(int argc, char* argv[])
{

  int devices = backend::get_dev_count();
  for(int i=0; i<devices; i++)
  {
    backend::set_dev(i);
    std::size_t free = backend::get_free_mem();
    std::size_t total = backend::get_total_mem();
    printf("device id %d total %.2fMB free %.2fMB (%.2f%%)\n",
      i, (double)total/1024./1024., (double)free/1024./1024.,
      (double)free/(double)total*100);
  }

  return 0;
}
