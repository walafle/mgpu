// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE core.invoke

#include <boost/test/unit_test.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <mgpu/environment.hpp>
#include <mgpu/invoke.hpp>
#include <mgpu/backend/backend.hpp>
#include <mgpu/exception.hpp>

using namespace mgpu;

void simple_callable(unsigned int * v, boost::mutex * m)
{
  // multithreaded access to v, protect with mutex
  boost::mutex::scoped_lock sl(*m);
  (*v)++;
  return;
}

void exception_callable()
{
}

// simple
//______________________________________________________________________________
BOOST_AUTO_TEST_CASE(simple)
{
  int devices = 0;
  unsigned int x1 = 0, x2 = 0;
  {
    boost::mutex m1, m2;
    environment env;
    devices = environment::get_size();

    // no arguments
    invoke_all(exception_callable);
    invoke(exception_callable, 0);

    invoke_all(simple_callable, &x1, &m1);
    for(int dev=0; dev<devices; dev++)
    {
      invoke(simple_callable, &x2, &m2, dev);
    }
  }
  BOOST_CHECK_EQUAL(x1, x2);
  BOOST_CHECK_EQUAL(x1, devices);
}

// exception
//______________________________________________________________________________
BOOST_AUTO_TEST_CASE(exception)
{
  int devices = environment::get_size();
  environment env;

  BOOST_REQUIRE_THROW(invoke(exception_callable, devices),
    mgpu_exception);
}
