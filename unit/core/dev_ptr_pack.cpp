// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE core.dev_ptr_pack

#include <boost/test/unit_test.hpp>

#include <mgpu/core/dev_ptr_pack.hpp>
#include <mgpu/core/dev_ptr.hpp>
#include <mgpu/backend/backend.hpp>

using namespace mgpu;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic)
{
  int dummy;

  dev_ptr_pack<int> p0;
  BOOST_CHECK_EQUAL(p0.size(), 0);

  int * ptr = &dummy;
  dev_ptr<int> dptr(ptr, 0);

  p0[0] = dptr;
  BOOST_CHECK(p0[0].get_raw_pointer() == ptr);

#if MGPU_NUM_DEVICES > 1
  p0[1] = dptr;
  BOOST_CHECK(p0[1].get_raw_pointer() == ptr);
#endif

  p0.set_size_(5);
  BOOST_CHECK_EQUAL(p0.size(), 5);
}

// copy
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(copy)
{
  int dummy;
  int * dummy_ptr = &dummy;

  dev_ptr<int> dummy_dev_ptr(dummy_ptr, 0);

  dev_ptr_pack<int> p0;
  dev_ptr_pack<int> p1;

  p0[0] = dummy_dev_ptr;
  p0.set_size_(5);
  p1 = p0;

  BOOST_CHECK(p0[0] == p1[0]);
  BOOST_CHECK_EQUAL(p0.size(), p1.size());
}
