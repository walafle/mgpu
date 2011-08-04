// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE container.dev_memory

#include <algorithm>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <test_types.hpp>
#include <sequences.hpp>

#include <mgpu/backend/backend.hpp>
#include <mgpu/core/dev_ptr.hpp>

using namespace mgpu;
using namespace mgpu::unit;
using namespace mgpu::backend;


// transfer
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(transfer, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  std::vector<T> in(num);
  std::generate(in.begin(), in.end(), rising_sequence<T>());

  std::vector<T> out(num);

  dev_ptr<T> dp1;
  dev_ptr<T> dp2;

  BOOST_REQUIRE_NO_THROW(dp1 = dev_malloc<T>(num));
  BOOST_REQUIRE_NO_THROW(dp2 = dev_malloc<T>(num));

  BOOST_REQUIRE_NO_THROW(copy(&in[0], dp1, num));
  BOOST_REQUIRE_NO_THROW(mgpu::backend::copy(dp1, dp2, num));
  BOOST_REQUIRE_NO_THROW(copy(dp2, &out[0], num));

  BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()) == true);

  BOOST_REQUIRE_NO_THROW(dev_free(dp1));
  BOOST_REQUIRE_NO_THROW(dev_free(dp2));
}


// distributed
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(distributed)
{
  const int num = 128;
  typedef float T;

  int devices = 0;
  BOOST_REQUIRE_NO_THROW(devices = get_dev_count());

  if(devices < 1)
  {
    return;
  }

  BOOST_REQUIRE_NO_THROW(set_dev(0));
  dev_ptr<T> dp;
  BOOST_REQUIRE_NO_THROW(dp = dev_malloc<T>(num));
  BOOST_CHECK_EQUAL(0, dp.dev_id());
  BOOST_REQUIRE_NO_THROW(dev_free(dp));

  if(devices < 2)
  {
    return;
  }

  BOOST_REQUIRE_NO_THROW(set_dev(1));
  BOOST_REQUIRE_NO_THROW(dp = dev_malloc<T>(num));
  BOOST_CHECK_EQUAL(1, dp.dev_id());
  BOOST_REQUIRE_NO_THROW(dev_free(dp));
}
