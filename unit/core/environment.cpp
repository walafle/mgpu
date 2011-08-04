// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE core.environment

#include <boost/test/unit_test.hpp>

#include <test_types.hpp>

#include <mgpu/environment.hpp>
#include <mgpu/backend/backend.hpp>

using namespace mgpu;
using namespace mgpu::unit;

// compile
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(compile)
{
  environment e;
}

// different_sizes
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(different_sizes)
{
#if MGPU_NUM_DEVICES > 1
  {
    environment e(dev_group(0, 1));
  }
#endif
#if MGPU_NUM_DEVICES > 2
  {
    environment e(dev_group(0, 1, 2));
  }
#endif
#if MGPU_NUM_DEVICES > 3
  {
    environment e(dev_group(0, 1, 2, 3));
  }
#endif
#if MGPU_NUM_DEVICES > 4
  {
    environment e(dev_group(0, 1, 2, 3, 4));
  }
#endif
#if MGPU_NUM_DEVICES > 5
  {
    environment e(dev_group(0, 1, 2, 3, 4, 5));
  }
#endif
#if MGPU_NUM_DEVICES > 6
  {
    environment e(dev_group(0, 1, 2, 3, 4, 5, 6));
  }
#endif
#if MGPU_NUM_DEVICES > 7
  {
    environment e(dev_group(0, 1, 2, 3, 4, 5, 6, 7));
  }
#endif
}

// rank
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(rank)
{
  int devices = MGPU_NUM_DEVICES;
  dev_group g = dev_group::from_to(0, devices);
  environment e(g);
  dev_group r = environment::get_all_ranks();
  BOOST_CHECK_EQUAL(r.size(), devices);
  for(int i=0; i<devices; ++i)
  {
    BOOST_CHECK_EQUAL(i, r[i]);
  }
}
