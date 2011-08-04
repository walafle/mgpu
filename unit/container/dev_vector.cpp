// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE container.dev_vector
#define MGPU_ALLOW_IMPLICIT_SCOPE_CHANGES

#include <boost/test/unit_test.hpp>

#include <mgpu/container/dev_vector.hpp>
#include <mgpu/transfer/copy.hpp>

#include <test_types.hpp>
#include <sequences.hpp>

using namespace mgpu;
using namespace mgpu::unit;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(basic, Type, test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  std::vector<T> in(num);
  std::generate(in.begin(), in.end(), rising_sequence<T>());

  std::vector<T> out(num, T(0));

  dev_vector<T> dev_vec1(num);
  dev_vector<T> dev_vec2(num);

  copy(in, dev_vec1.begin());
  copy(dev_vec1, dev_vec2.begin());
  copy(dev_vec2, out.begin());

  BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()) == true);

}

// distributed
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(distributed, Type, test_types_and_sizes)
{

  int devices = MGPU_NUM_DEVICES;
  if(devices < 2)
  {
    return;
  }

  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  std::vector<T> in(num);
  std::generate(in.begin(), in.end(), rising_sequence<T>());

  std::vector<T> out(num, T(0));

  backend::set_dev(0);
  dev_vector<T> dev_vec1(num);
  backend::set_dev(1);
  dev_vector<T> dev_vec2(num);

  copy(in, dev_vec1.begin());
  copy(dev_vec1, dev_vec2.begin());
  copy(dev_vec2, out.begin());

  BOOST_CHECK(std::equal(in.begin(), in.end(), out.begin()) == true);

}
