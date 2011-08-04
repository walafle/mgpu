// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_UNIT_TEST_TYPES_HPP
#define MGPU_UNIT_TEST_TYPES_HPP

/**
 * @file utils.hpp
 *
 * This header provides utilities for unit tests
 */

#include <complex>

#include <boost/mpl/fold.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/vector_c.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/placeholders.hpp>

#include <mgpu/config.hpp>

namespace mgpu
{

namespace unit
{


template<class S1, class S2> struct cartesian_product_types
{
  template<class V, class S, class State>
  struct inner
  {
    typedef typename boost::mpl::fold<S
                             , State
                             , boost::mpl::push_back< boost::mpl::_1
                                                    , std::pair<V
                                                               , boost::mpl::_2
                                                               >
                                                    >
                             >::type type;
  };

  typedef typename boost::mpl::fold<S1
                                   , boost::mpl::vector<>
                                   , inner<boost::mpl::_2, S2, boost::mpl::_1>
                                   >::type type;
};

// generic test types
typedef boost::mpl::vector_c<int, 1, 3, 1024, 1024*16> sizes;

typedef boost::mpl::vector<char, float, double,
                           std::complex<float>, std::complex<double> > types;

typedef cartesian_product_types<sizes, types>::type test_types_and_sizes;



typedef boost::mpl::vector<char, float, double,
                            std::complex<float>, std::complex<double> >
test_types;

typedef boost::mpl::vector<char, unsigned int, float, std::complex<float> >
test_types_no_double;

typedef boost::mpl::vector<char, short int, unsigned int, int, long int>
test_types_no_float;


// test types for numeric computation
#ifdef MGPU_DEVICE_DOUBLE_SUPPORT

typedef boost::mpl::vector<float, double,
                           std::complex<float>, std::complex<double> >
numeric_test_types;
typedef boost::mpl::vector<std::complex<float>, std::complex<double> >
numeric_test_types_complex;

typedef boost::mpl::vector<
  boost::mpl::vector<float, std::complex<float> >,
  boost::mpl::vector<std::complex<float>, float>,
  boost::mpl::vector<std::complex<float>, std::complex<float> >,
  boost::mpl::vector<double, std::complex<double> >,
  boost::mpl::vector<std::complex<double>, double>,
  boost::mpl::vector<std::complex<double>, std::complex<double> >
> fft_test_types;

#else

typedef boost::mpl::vector<float, std::complex<float> >
numeric_test_types;
typedef boost::mpl::vector<std::complex<float> >
numeric_test_types_complex;

typedef boost::mpl::vector<
  boost::mpl::vector<float, std::complex<float> >,
  boost::mpl::vector<std::complex<float>, float>,
  boost::mpl::vector<std::complex<float>, std::complex<float> >
> fft_test_types;

#endif


typedef cartesian_product_types<sizes, numeric_test_types>::type
numeric_test_types_and_sizes;

typedef cartesian_product_types<sizes, numeric_test_types_complex>::type
numeric_test_types_complex_and_sizes;


} // namespace unit

} // namespace mgpu


#endif // MGPU_UNIT_TEST_TYPES_HPP
