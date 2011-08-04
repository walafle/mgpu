// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE core.synchronization

#include <boost/test/unit_test.hpp>

#include <test_types.hpp>

#include <mgpu/environment.hpp>
#include <mgpu/synchronization.hpp>

using namespace mgpu;
using namespace mgpu::unit;

// compile
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE(compile)
{
  environment e;
  barrier();
  synchronize();
  synchronize_barrier();

//  barrier(non_blocking);
//  synchronize_barrier(non_blocking);
}

