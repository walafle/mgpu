// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE core.invoke_kernel


#include <boost/test/unit_test.hpp>

#include <mgpu/invoke_kernel.hpp>
#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/synchronization.hpp>

#include <test_types.hpp>

using namespace mgpu;
using namespace mgpu::unit;

typedef environment env;

void kernel_caller0()
{
}

template <typename T>
void kernel_caller1(dev_range<T> vec1, bool & set, int & size)
{
  if(set)
  {
    size = vec1.size();
  }
}

template <typename T>
void kernel_caller2(dev_range<T> vec1, dev_range<T> vec2, int & i)
{
}

void kernel_caller3(dev_id_t id, dev_id_t & retval)
{
  retval = id;
}

template <typename T>
void kernel_caller4(seg_dev_vector<T> & vec)
{
}

BOOST_GLOBAL_FIXTURE(environment);

// basic
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE_TEMPLATE(basic, T, types)
{
  int devices = environment::get_size();

  seg_dev_vector<T> dev_vec1(devices);
  seg_dev_vector<T> dev_vec2(devices);

  invoke_kernel_all(kernel_caller0);
  invoke_kernel(kernel_caller0, devices-1);

  int size = 0;
  bool update_size = false;
  invoke_kernel_all(kernel_caller1<T>, dev_vec1, update_size, size);

  update_size = true;
  invoke_kernel(kernel_caller1<T>, dev_vec1, update_size, size, devices-1);

  // wait for all operations in the queues to finish
  barrier();
  BOOST_CHECK_EQUAL(size, 1);

  invoke_kernel_all(kernel_caller2<T>, dev_vec1, dev_vec2, size);
  invoke_kernel(kernel_caller2<T>, dev_vec1, dev_vec2, size, devices-1);

}

// dev_id_special
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE_TEMPLATE(dev_id_special, T, types)
{
  int devices = environment::get_size();

  dev_id_t retval = -1;
  invoke_kernel(kernel_caller3, pass_dev_id, retval, devices-1);

  barrier();
  BOOST_CHECK_EQUAL(retval, devices-1);
}

// pass_through_special
// _____________________________________________________________________________
BOOST_AUTO_TEST_CASE_TEMPLATE(pass_through_special, T, types)
{
  int devices = environment::get_size();

  seg_dev_vector<T> dev_vec(devices);
  pass_through<seg_dev_vector<T> > pt(dev_vec);
  invoke_kernel(kernel_caller4<T>, pt, devices-1);

}
