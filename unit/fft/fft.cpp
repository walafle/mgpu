// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE fft.fft

#include <boost/test/unit_test.hpp>

#include <mgpu/fft.hpp>

#include <test_types.hpp>

using namespace mgpu;
using namespace mgpu::unit;

BOOST_GLOBAL_FIXTURE(environment);

// fftclass
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(fftclass)
{
  fft<float, std::complex<float> > f(128, 128);
}

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
