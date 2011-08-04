// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE backend.management

#include <boost/test/unit_test.hpp>

#include <mgpu/backend/backend.hpp>

using namespace mgpu;
using namespace mgpu::backend;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic)
{
  int devices = 0;
  BOOST_REQUIRE_NO_THROW(devices = get_dev_count());

  if(devices < 1)
  {
    return;
  }

  BOOST_REQUIRE_NO_THROW(sync_dev());
  BOOST_REQUIRE_NO_THROW(set_dev(0));
  BOOST_REQUIRE_NO_THROW(reset_dev());
}

// p2p
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(p2p)
{
  int devices = 0;
  BOOST_REQUIRE_NO_THROW(devices = get_dev_count());

  if(devices < 2)
  {
    return;
  }

  bool p2p_possible_0_1 = false;
  bool p2p_possible_1_0 = false;
  BOOST_REQUIRE_NO_THROW(p2p_possible_0_1 = p2p_possible(0, 1));
  BOOST_REQUIRE_NO_THROW(p2p_possible_1_0 = p2p_possible(1, 0));

  if(p2p_possible_0_1)
  {
    BOOST_REQUIRE_NO_THROW(set_dev(0));
    BOOST_REQUIRE_NO_THROW(enable_p2p(1));
    BOOST_REQUIRE_NO_THROW(disable_p2p(1));
  }
  if(p2p_possible_1_0)
  {
    BOOST_REQUIRE_NO_THROW(set_dev(1));
    BOOST_REQUIRE_NO_THROW(enable_p2p(0));
    BOOST_REQUIRE_NO_THROW(disable_p2p(0));
  }

  BOOST_REQUIRE_NO_THROW(set_dev(0));
  BOOST_REQUIRE_NO_THROW(reset_dev());
  BOOST_REQUIRE_NO_THROW(set_dev(1));
  BOOST_REQUIRE_NO_THROW(reset_dev());
}

// exception
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(exception)
{
  int devices = 0;
  BOOST_REQUIRE_NO_THROW(devices = get_dev_count());
  BOOST_REQUIRE_THROW(set_dev(devices), device_exception);

}
