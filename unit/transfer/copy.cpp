// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE transfer.copy

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


// seg_dev_seg_dev
// copy from segmented device memory to segmented device memory
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(seg_dev_seg_dev, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  {
    seg_dev_vector<T> vec1(num*devices);
    seg_dev_vector<T> vec2(num*devices);
    std::vector<T> in(num*devices);
    std::vector<T> out(num*devices);
    std::generate(in.begin(), in.end(), rising_sequence<T>());

    copy(in, vec1.begin());
    copy(vec1, vec2.begin());
    copy(vec2, out.begin());

    synchronize_barrier();
    BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()) == true);
  }
  {
    seg_dev_stream s;
    seg_dev_vector<T> vec1(num*devices);
    seg_dev_vector<T> vec2(num*devices);
    std::vector<T> in(num*devices);
    std::vector<T> out(num*devices);
    std::generate(in.begin(), in.end(), rising_sequence<T>());

    copy(in, vec1.begin());
    copy(vec1, vec2.begin(), s);
    copy(vec2, out.begin());

    synchronize_barrier();
    BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()) == true);
  }
}


// seg_dev_dev
// copy from segmented device memory to device memory
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(seg_dev_dev, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  {
    seg_dev_vector<T> vec1(num*devices);
    dev_vector<T> vec2(num*devices);
    std::vector<T> in(num*devices);
    std::vector<T> out(num*devices);
    std::generate(in.begin(), in.end(), rising_sequence<T>());

    copy(in, vec1.begin());
    copy(vec1, vec2.begin());
    synchronize_barrier();
    copy(vec2, out.begin());

    BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()) == true);
  }
  {
    seg_dev_stream s;
    seg_dev_vector<T> vec1(num*devices);
    dev_vector<T> vec2(num*devices);
    std::vector<T> in(num*devices);
    std::vector<T> out(num*devices);
    std::generate(in.begin(), in.end(), rising_sequence<T>());

    copy(in, vec1.begin());
    copy(vec1, vec2.begin(), s);
    synchronize_barrier();
    copy(vec2, out.begin());

    BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()) == true);
  }
}



// dev_seg_dev
// copy from device to segmented device
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(dev_seg_dev, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  {
    dev_vector<T> vec1(num*devices);
    seg_dev_vector<T> vec2(num*devices);
    std::vector<T> in(num*devices);
    std::vector<T> out(num*devices);
    std::generate(in.begin(), in.end(), rising_sequence<T>());

    copy(in, vec1.begin());
    copy(vec1, vec2.begin());
    copy(vec2, out.begin());

    synchronize_barrier();
    BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()) == true);
  }
  {
    seg_dev_stream s;
    dev_vector<T> vec1(num*devices);
    seg_dev_vector<T> vec2(num*devices);
    std::vector<T> in(num*devices);
    std::vector<T> out(num*devices);
    std::generate(in.begin(), in.end(), rising_sequence<T>());

    copy(in, vec1.begin());
    copy(vec1, vec2.begin(), s);
    copy(vec2, out.begin());

    synchronize_barrier();
    BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()) == true);
  }
}

// dev_dev
// copy from device to device
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(dev_dev, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  backend::dev_stream s;
  dev_vector<T> vec1(num*devices);
  dev_vector<T> vec2(num*devices);

  std::vector<T> in(num*devices);
  std::vector<T> out(num*devices);

  std::generate(in.begin(), in.end(), rising_sequence<T>());
  copy(in, vec1.begin());

  copy(vec1, vec2.begin());
  copy(vec2, out.begin());
  BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()) == true);

  out.assign(out.size(), T(0));
  copy(out, vec2.begin());
  copy(vec1, vec2.begin(), s);

  backend::sync_dev();
  copy(vec2, out.begin());
  BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()) == true);
}

// host_seg_dev_reverse
// copy from host to segmented device
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(host_seg_dev_reverse, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  {
    std::vector<T> vec1(num*devices);
    seg_dev_vector<T> vec2(num*devices);
    std::vector<T> out(num*devices);
    std::generate(vec1.begin(), vec1.end(), rising_sequence<T>());

    copy(vec1, vec2.begin());
    copy(vec2, out.begin());
    synchronize_barrier();
    BOOST_CHECK(std::equal(vec1.begin(), vec1.end(), out.begin()) == true);
  }
  {
    seg_dev_stream s;
    std::vector<T> vec1(num*devices);
    seg_dev_vector<T> vec2(num*devices);
    std::vector<T> out(num*devices);
    std::generate(vec1.begin(), vec1.end(), rising_sequence<T>());

    copy(vec1, vec2.begin(), s);
    copy(vec2, out.begin());
    synchronize_barrier();
    BOOST_CHECK(std::equal(vec1.begin(), vec1.end(), out.begin()) == true);
  }
}


// host_dev_dev_host
// copy from host to device and back
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(host_dev_dev_host, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  backend::dev_stream s;
  std::vector<T> vec1(num*devices);
  dev_vector<T> vec2(num*devices);
  std::vector<T> out(num*devices);

  std::generate(vec1.begin(), vec1.end(), rising_sequence<T>());
  copy(vec1, vec2.begin());
  copy(vec2, out.begin());
  BOOST_CHECK(std::equal(vec1.begin(), vec1.end(), out.begin()) == true);

  out.assign(vec2.size(), T(0));

  copy(vec1, vec2.begin(), s);
  backend::sync_dev();
  copy(vec2, out.begin());
  BOOST_CHECK(std::equal(vec1.begin(), vec1.end(), out.begin()) == true);
}

// host_host
// copy on host
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(host_host, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  backend::dev_stream s;
  std::vector<T> vec1(num*devices);
  std::vector<T> vec2(num*devices);

  std::generate(vec1.begin(), vec1.end(), rising_sequence<T>());
  copy(vec1, vec2.begin());
  BOOST_CHECK(std::equal(vec1.begin(), vec1.end(), vec2.begin()) == true);

  vec2.assign(vec2.size(), T(0));

  copy(vec1, vec2.begin(), s);
  BOOST_CHECK(std::equal(vec1.begin(), vec1.end(), vec2.begin()) == true);
}
