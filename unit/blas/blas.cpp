// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE blas.blas

#include <boost/test/unit_test.hpp>

#include <mgpu/blas.hpp>

using namespace mgpu;

BOOST_GLOBAL_FIXTURE(environment);

// blasclass
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(blasclass)
{
  blas b;
  b.set_scalar_device();
  b.set_scalar_host();
}

