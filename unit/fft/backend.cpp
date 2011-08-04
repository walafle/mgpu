// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE fft.backend

#include <boost/test/unit_test.hpp>
#include <boost/mpl/at.hpp>

#include <mgpu/backend/fft.hpp>
#include <mgpu/transfer/copy.hpp>

#include <test_types.hpp>
#include <sequences.hpp>

using namespace mgpu;
using namespace mgpu::backend;
using namespace mgpu::unit;

// creation
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(creation, T, fft_test_types)
{
  typedef typename boost::mpl::at_c<T, 0>::type T1;
  typedef typename boost::mpl::at_c<T, 1>::type T2;

  {
    fft<T1, T2> plan0 (256, 256);
    fft<T1, T2> plan1 (384, 384);
    fft<T1, T2> plan2 (512, 512);
  }

  {
    fft<T1, T2> plan0 (256, 256, 4);
    fft<T1, T2> plan1 (384, 384, 4);
    fft<T1, T2> plan2 (512, 512, 4);
  }

}

// operations
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(operations, T, fft_test_types)
{
  typedef typename boost::mpl::at_c<T, 0>::type T1;
  typedef typename boost::mpl::at_c<T, 1>::type T2;

  std::vector<T1> A_(256*256);
  std::vector<T2> B_(256*256);

  std::generate(A_.begin(), A_.end(), random_sequence<T1>());

  dev_vector<T1> A(256*256);
  dev_vector<T2> B(256*256);

  copy(A_, A.begin());

  fft<T1, T2> plan (256, 256);
  if(plan.forward_possible)
  {
    plan.forward(A, B);
  }
  if(plan.inverse_possible)
  {
    plan.inverse(A, B);
  }

}
