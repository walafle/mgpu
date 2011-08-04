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
  dev_stream s;
  BOOST_REQUIRE_NO_THROW(sync_dev(s));
}

