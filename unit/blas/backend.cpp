// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----


//
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//


#define BOOST_TEST_MODULE blas.backend

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <mgpu/backend/blas.hpp>
#include <mgpu/transfer/copy.hpp>

#include <test_types.hpp>
#include <sequences.hpp>
#include <helper_functions.hpp>

using namespace mgpu;
using namespace mgpu::unit;
using namespace mgpu::backend;
namespace ublas = boost::numeric::ublas;

// helper struct
template <typename T, std::size_t N>
struct blas_backend_data
{
    blas_backend_data() :
      X_(N), Y_(N), alpha_(1), r1_(0), r2_(0),
      X(N), Y(N), alpha(1), r1(1), r2(1),
      r__(0), Y__(N)
    {
      std::generate(X_.begin(), X_.end(), random_sequence<T>());
      std::generate(Y_.begin(), Y_.end(), random_sequence<T>());
    };

    ublas::vector<T> X_;
    ublas::vector<T> Y_;
    ublas::vector<T> alpha_;
    T r1_;
    T r2_;

    dev_vector<T> X;
    dev_vector<T> Y;
    dev_vector<T> alpha;
    dev_vector<T> r1;
    dev_vector<T> r2;

    // gold results
    T r__;
    ublas::vector<T> Y__;
};


// blasclass
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE(blasclass)
{
  blas b;
}

// inner_prod
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(inner_prod, Type, numeric_test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  blas_backend_data<T, num> d;
  blas b;

  // upload X and y
  copy(d.X_, d.X.begin());
  copy(d.Y_, d.Y.begin());

  // do calculation
  b.set_scalar_host();
  b.inner_prod(d.X, d.Y, &d.r1_);
  b.set_scalar_device();
  b.inner_prod(d.X, d.Y, d.r2.begin());

  copy(d.r2, &d.r2_);

  // calculate gold value
  d.r__ = ublas::inner_prod(d.X_, d.Y_);

  // test host and device result
  BOOST_CHECK_CLOSE(std::abs(d.r__), std::abs(d.r1_), 0.00001);
  BOOST_CHECK_CLOSE(std::abs(d.r__), std::abs(d.r2_), 0.00001);
}

// inner_prod_c
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(inner_prod_c, Type,
  numeric_test_types_complex_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  blas_backend_data<T, num> d;
  blas b;

  // upload X and y
  copy(d.X_, d.X.begin());
  copy(d.Y_, d.Y.begin());

  // do calculation
  b.set_scalar_host();
  b.inner_prod_c(d.X, d.Y, &d.r1_);
  b.set_scalar_device();
  b.inner_prod_c(d.X, d.Y, d.r2.begin());

  copy(d.r2, &d.r2_);

  // calculate gold value
  d.r__ = ublas::inner_prod(ublas::conj(d.X_), d.Y_);

  // test host and device result
  BOOST_CHECK_CLOSE(std::abs(d.r__), std::abs(d.r1_), 0.00001);
  BOOST_CHECK_CLOSE(std::abs(d.r__), std::abs(d.r2_), 0.00001);
}

// axpy
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(axpy, Type, numeric_test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  blas_backend_data<T, num> d;
  blas b;

  d.Y__ = d.Y_;

  dev_vector<T> Y1(num);
  dev_vector<T> Y2(num);

  copy(d.X_, d.X.begin());
  copy(d.Y_, Y1.begin());
  copy(d.Y_, Y2.begin());

  d.alpha_[0] = make_number(2.5);
  copy(d.alpha_, d.alpha.begin());

  // device calculation
  b.set_scalar_device();
  b.axpy(d.alpha.begin(), d.X, Y1);
  b.set_scalar_host();
  b.axpy(&(d.alpha_[0]), d.X, Y2);

  // host calculation
  d.Y__ += d.alpha_[0] * d.X_;

  // test host and device result
  copy(Y1, d.Y_.begin());
  for(int i=0; i<num; ++i)
  {
    BOOST_CHECK_CLOSE(std::abs(d.Y_[i]), std::abs(d.Y__[i]), 0.00001);
  }
  copy(Y2, d.Y_.begin());
  for(int i=0; i<num; ++i)
  {
    BOOST_CHECK_CLOSE(std::abs(d.Y_[i]), std::abs(d.Y__[i]), 0.00001);
  }
}
