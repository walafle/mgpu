// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE transfer.gather

#include <algorithm>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <test_types.hpp>
#include <sequences.hpp>

#include <mgpu/environment.hpp>
#include <mgpu/synchronization.hpp>
#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/transfer/gather.hpp>
#include <mgpu/transfer/broadcast.hpp>
#include <mgpu/transfer/copy.hpp>

using namespace mgpu;
using namespace mgpu::unit;
using namespace mgpu::backend;


BOOST_GLOBAL_FIXTURE(environment);

// host_device
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(host_device, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  seg_dev_vector<T> dist(num*devices);

  std::vector<T> in(num);
  std::vector<T> out(num*devices);

  std::generate(in.begin(), in.end(), rising_sequence<T>());

  broadcast(in, dist.begin());
  gather(dist, out.begin());

  // make sure the copies were started and they finished
  synchronize_barrier();

  for(std::size_t i=0; i<dist.segments(); i++)
  {
    BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()+num*i) == true);
  }
}

// device_device
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(device_device, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  dev_vector<T> vec(num*devices);
  seg_dev_vector<T> dist(num*devices);

  std::vector<T> in(num);
  std::vector<T> out(num*devices);

  std::generate(in.begin(), in.end(), rising_sequence<T>());

  broadcast(in, dist.begin());
  gather(dist, vec.begin());

  // make sure the copies were started and they finished
  synchronize_barrier();

  copy(vec, out.begin());
  synchronize_barrier();

  for(std::size_t i=0; i<dist.segments(); i++)
  {
    BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()+num*i) == true);
  }
}

// pointer
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(pointer, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  seg_dev_vector<T> dist(num*devices);

  std::vector<T> in(num);
  std::vector<T> out(num*devices);

  T * in_ptr = &in[0];
  T * out_ptr = &out[0];

  std::generate(in.begin(), in.end(), rising_sequence<T>());

  broadcast(make_range(in_ptr, in_ptr+num), dist.begin());
  gather(dist, out_ptr);

  // make sure the copies were started and they finished
  synchronize_barrier();


  for(std::size_t i=0; i<dist.segments(); i++)
  {
    BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()+num*i) == true);
  }
}
