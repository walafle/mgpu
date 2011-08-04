// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE container.seg_dev_vector

#include <algorithm>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <test_types.hpp>
#include <sequences.hpp>

#include <mgpu/environment.hpp>
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

  BOOST_CHECK_EQUAL(vec.size(), num*devices);
  BOOST_CHECK_EQUAL(vec.blocksize(), 1);
  BOOST_CHECK_EQUAL(vec.blocks(), num*devices);
  BOOST_CHECK_EQUAL(vec.segments(), devices);

  for(int i=0; i<devices; i++)
  {
    BOOST_CHECK_EQUAL(vec.blocks(i), num);
    BOOST_CHECK_EQUAL(vec.size(i), num);
  }
}

// partial - using only some devices not all
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(partial, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size() / 2;
  if(devices < 1)
  {
    devices = environment::get_size();
  }

  rank_group r = rank_group::from_to(0, devices);

  seg_dev_vector<T> vec(num*devices, r);

  BOOST_CHECK_EQUAL(vec.size(), num*devices);
  BOOST_CHECK_EQUAL(vec.blocksize(), 1);
  BOOST_CHECK_EQUAL(vec.blocks(), num*devices);
  BOOST_CHECK_EQUAL(vec.segments(), devices);

  for(int i=0; i<devices; i++)
  {
    BOOST_CHECK_EQUAL(vec.blocks(i), num);
    BOOST_CHECK_EQUAL(vec.size(i), num);
  }
}

// blocked_basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(blocked_basic, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;
  const int blocksize = 16;

  int devices = environment::get_size();

  seg_dev_vector<T> vec(num*devices*blocksize, blocksize);

  BOOST_CHECK_EQUAL(vec.size(), num*devices*blocksize);
  BOOST_CHECK_EQUAL(vec.blocksize(), blocksize);
  BOOST_CHECK_EQUAL(vec.blocks(), num*devices);
  BOOST_CHECK_EQUAL(vec.segments(), devices);

  for(int i=0; i<devices; i++)
  {
    BOOST_CHECK_EQUAL(vec.blocks(i), num);
    BOOST_CHECK_EQUAL(vec.size(i), num*blocksize);
  }
}

// blocked_one_missing
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(blocked_one_missing)
{
  int devices = environment::get_size();
  int devices_minus_one = devices - 1;
  if(devices < 2)
  {
    return;
  }

  // we request N-1 blocks for N devices
  const int blocksize = 16;
  const int num = blocksize * devices_minus_one;
  typedef float T;

  seg_dev_vector<T> vec(num, blocksize);

  BOOST_CHECK_EQUAL(vec.size(), num);
  BOOST_CHECK_EQUAL(vec.blocksize(), blocksize);
  BOOST_CHECK_EQUAL(vec.blocks(), devices_minus_one);
  BOOST_CHECK_EQUAL(vec.segments(), devices_minus_one);

  for(int i=0; i<devices_minus_one; i++)
  {
    BOOST_CHECK_EQUAL(vec.blocks(i), 1);
    BOOST_CHECK_EQUAL(vec.size(i), blocksize);
  }
  BOOST_CHECK_EQUAL(vec.blocks(devices_minus_one), 0);
  BOOST_CHECK_EQUAL(vec.size(devices_minus_one), 0);
}

// blocked_one_more
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(blocked_one_more)
{
  int devices = environment::get_size();
  int devices_plus_one = devices + 1;

  // we request N-1 blocks for N devices
  const int blocksize = 16;
  const int num = blocksize * devices_plus_one;
  typedef float T;

  seg_dev_vector<T> vec(num, blocksize);

  BOOST_CHECK_EQUAL(vec.size(), num);
  BOOST_CHECK_EQUAL(vec.blocksize(), blocksize);
  BOOST_CHECK_EQUAL(vec.blocks(), devices_plus_one);
  BOOST_CHECK_EQUAL(vec.segments(), devices);

  BOOST_CHECK_EQUAL(vec.blocks(0), 2);
  BOOST_CHECK_EQUAL(vec.size(0), 2*blocksize);

  for(int i=1; i<devices; i++)
  {
    BOOST_CHECK_EQUAL(vec.blocks(i), 1);
    BOOST_CHECK_EQUAL(vec.size(i), blocksize);
  }
}

// named_ctors
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(named_ctors, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  seg_dev_vector<T> vec = seg_dev_vector<T>::split(num*devices);

  BOOST_CHECK_EQUAL(vec.size(), num*devices);
  BOOST_CHECK_EQUAL(vec.blocksize(), 1);
  BOOST_CHECK_EQUAL(vec.blocks(), num*devices);
  BOOST_CHECK_EQUAL(vec.segments(), devices);

  for(int i=0; i<devices; i++)
  {
    BOOST_CHECK_EQUAL(vec.blocks(i), num);
    BOOST_CHECK_EQUAL(vec.size(i), num);
  }


  const int blocksize = 16;

  seg_dev_vector<T> vecb =
    seg_dev_vector<T>::split(num*devices*blocksize, blocksize);

  BOOST_CHECK_EQUAL(vecb.size(), num*devices*blocksize);
  BOOST_CHECK_EQUAL(vecb.blocksize(), blocksize);
  BOOST_CHECK_EQUAL(vecb.blocks(), num*devices);
  BOOST_CHECK_EQUAL(vecb.segments(), devices);

  for(int i=0; i<devices; i++)
  {
    BOOST_CHECK_EQUAL(vecb.blocks(i), num);
    BOOST_CHECK_EQUAL(vecb.size(i), num*blocksize);
  }

  seg_dev_vector<T> vecc =
    seg_dev_vector<T>::clone(num);

  BOOST_CHECK_EQUAL(vecc.size(), num*devices);
  BOOST_CHECK_EQUAL(vecc.blocksize(), 1);
  BOOST_CHECK_EQUAL(vecc.blocks(), num*devices);
  BOOST_CHECK_EQUAL(vecc.segments(), devices);

  for(int i=0; i<devices; i++)
  {
    BOOST_CHECK_EQUAL(vecc.blocks(i), num);
    BOOST_CHECK_EQUAL(vecc.size(i), num);
  }

  seg_dev_vector<T> vecc2((clone_size(num)));

  BOOST_CHECK_EQUAL(vecc2.size(), num*devices);
  BOOST_CHECK_EQUAL(vecc2.blocksize(), 1);
  BOOST_CHECK_EQUAL(vecc2.blocks(), num*devices);
  BOOST_CHECK_EQUAL(vecc2.segments(), devices);

  for(int i=0; i<devices; i++)
  {
    BOOST_CHECK_EQUAL(vecc2.blocks(i), num);
    BOOST_CHECK_EQUAL(vecc2.size(i), num);
  }
}

// basic_heap
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(basic_heap, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  seg_dev_vector<T> * vec = new seg_dev_vector<T>(num*devices);

  BOOST_CHECK_EQUAL(vec->size(), num*devices);
  BOOST_CHECK_EQUAL(vec->blocksize(), 1);
  BOOST_CHECK_EQUAL(vec->blocks(), num*devices);
  BOOST_CHECK_EQUAL(vec->segments(), devices);

  for(int i=0; i<devices; i++)
  {
    BOOST_CHECK_EQUAL(vec->blocks(i), num);
    BOOST_CHECK_EQUAL(vec->size(i), num);
  }

  synchronize_barrier();
  delete vec;
}
