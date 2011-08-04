// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE container.seg_dev_iterator

#include <algorithm>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <test_types.hpp>
#include <sequences.hpp>

#include <mgpu/environment.hpp>
#include <mgpu/synchronization.hpp>
#include <mgpu/container/seg_dev_vector.hpp>

using namespace mgpu;
using namespace mgpu::unit;
using namespace mgpu::backend;


BOOST_GLOBAL_FIXTURE(environment);

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(basic, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  seg_dev_vector<T> vec(num*devices);

  seg_dev_iterator<T> it1 = vec.begin();
  seg_dev_iterator<T> it2 = vec.begin();
  barrier();

  for(int i=0; i<devices; i++)
  {
    BOOST_CHECK(it1.get_pointer(i) == it2.get_pointer(i));
  }

  BOOST_CHECK(it1.get_pointer() == it2.get_pointer());
  BOOST_CHECK(it1 == it2);

  ++it1;

  for(int i=0; i<devices; i++)
  {
    BOOST_CHECK(it1.get_pointer(i) != it2.get_pointer(i));
  }

  BOOST_CHECK(it1.get_pointer() != it2.get_pointer());
  BOOST_CHECK(it1 != it2);

  ++it2;

  for(int i=0; i<devices; i++)
  {
    BOOST_CHECK(it1.get_pointer(i) == it2.get_pointer(i));
  }

  BOOST_CHECK(it1.get_pointer() == it2.get_pointer());
  BOOST_CHECK(it1 == it2);
}

//// begin_end
//// _____________________________________________________________________________
//
//BOOST_AUTO_TEST_CASE_TEMPLATE(begin_end, Type, test_types_and_sizes)
//{
//  const int num = Type::first_type::value;
//  typedef typename Type::second_type T;
//
//  int devices = backend::get_dev_count();
//
//  seg_dev_vector<T> vec(num*devices);
//
//  seg_dev_iterator<T> it1 = vec.begin();
//  seg_dev_iterator<T> it2 = vec.end();
//  barrier();
//
//  for(int i=0; i<devices; i++)
//  {
//    BOOST_CHECK(it1.raw(i) != it2.raw(i));
//  }
//
//  BOOST_CHECK(it1.raw() != it2.raw());
//  BOOST_CHECK(it1 != it2);
//
//  it1 += num*devices;
//
//  for(int i=0; i<devices; i++)
//  {
//    BOOST_CHECK(it1.raw(i) == it2.raw(i));
//  }
//
//  BOOST_CHECK(it1.raw() == it2.raw());
//  BOOST_CHECK(it1 == it2);
//
//}
