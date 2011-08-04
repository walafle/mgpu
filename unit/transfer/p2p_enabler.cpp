// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE transfer.p2p_nabler

#include <algorithm>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <mgpu/transfer/p2p_enabler.hpp>

using namespace mgpu;



BOOST_GLOBAL_FIXTURE(environment);

// host_device
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(host_device)
{
  p2p_enabler e(rank_group::from_to(0, 1));
}
