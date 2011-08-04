// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE core.dev_group

#include <boost/test/unit_test.hpp>

#include <mgpu/core/dev_group.hpp>
#include <mgpu/backend/backend.hpp>

using namespace mgpu;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic)
{
  dev_group d0;
  BOOST_CHECK_EQUAL(d0.size(), 0);

  dev_group d1(1);
  BOOST_CHECK_EQUAL(d1.size(), 1);
  BOOST_CHECK_EQUAL(d1[0], 1);

#if MGPU_NUM_DEVICES > 1
  dev_group d2(1, 2);
  BOOST_CHECK_EQUAL(d2.size(), 2);
  BOOST_CHECK_EQUAL(d2[0], 1);
  BOOST_CHECK_EQUAL(d2[1], 2);
#endif
#if MGPU_NUM_DEVICES > 2
  dev_group d3(1, 2, 4);
  BOOST_CHECK_EQUAL(d3.size(), 3);
  BOOST_CHECK_EQUAL(d3[0], 1);
  BOOST_CHECK_EQUAL(d3[1], 2);
  BOOST_CHECK_EQUAL(d3[2], 4);
#endif
}

// all_dev
// _____________________________________________________________________________

void all_dev_func(const dev_group & d)
{
  BOOST_CHECK_EQUAL(d.size(), MGPU_NUM_DEVICES);
  BOOST_CHECK_EQUAL(d[MGPU_NUM_DEVICES-1], MGPU_NUM_DEVICES-1);
}

BOOST_AUTO_TEST_CASE(all_dev)
{
  all_dev_func(all_devices);
}

// copy
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(copy)
{
  dev_group d1(1);
  d1 = all_devices;
  BOOST_CHECK_EQUAL(d1.size(), MGPU_NUM_DEVICES);
  BOOST_CHECK_EQUAL(d1[MGPU_NUM_DEVICES-1], MGPU_NUM_DEVICES-1);

  dev_group d2(all_devices);
  BOOST_CHECK_EQUAL(d2.size(), MGPU_NUM_DEVICES);
  BOOST_CHECK_EQUAL(d2[MGPU_NUM_DEVICES-1], MGPU_NUM_DEVICES-1);

}

// from_to
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(from_to)
{
  dev_group d1 = dev_group::from_to(0, MGPU_NUM_DEVICES);
  BOOST_CHECK_EQUAL(d1.size(), MGPU_NUM_DEVICES);
  BOOST_CHECK_EQUAL(d1[0], 0);
  BOOST_CHECK_EQUAL(d1[MGPU_NUM_DEVICES-1], MGPU_NUM_DEVICES-1);

  if(MGPU_NUM_DEVICES < 2)
  {
    return;
  }

  dev_group d2 = dev_group::from_to(1, MGPU_NUM_DEVICES);
  BOOST_CHECK_EQUAL(d2.size(), MGPU_NUM_DEVICES-1);
  BOOST_CHECK_EQUAL(d2[0], 1);
  BOOST_CHECK_EQUAL(d2[MGPU_NUM_DEVICES-2], MGPU_NUM_DEVICES-1);
}
