// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE core.ref

#include <boost/test/unit_test.hpp>

#include <mgpu/core/ref.hpp>

using namespace mgpu;

// basic
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(basic)
{
  int i = 1;
  int j = 2;

  int * ptr = NULL;
  int ** ind_ptr;

  ref<int *> ref(ptr);
  ind_ptr = ref.get_pointer();
  BOOST_CHECK_EQUAL(ref.get(), ptr);
  BOOST_CHECK_EQUAL(*ind_ptr, ptr);

  ptr = &i;
  BOOST_CHECK_EQUAL(ref.get(), &i);
  BOOST_CHECK_EQUAL(*ref.get(), i);
  BOOST_CHECK_EQUAL(*ind_ptr, ptr);

  ptr = &j;
  BOOST_CHECK_EQUAL(ref.get(), &j);
  BOOST_CHECK_EQUAL(*ref.get(), j);
  BOOST_CHECK_EQUAL(*ind_ptr, ptr);
}

