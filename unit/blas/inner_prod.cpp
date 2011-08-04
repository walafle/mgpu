// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE blas.seg_axpy

#include <numeric>

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <mgpu/blas.hpp>
#include <mgpu/transfer/scatter.hpp>
#include <mgpu/transfer/gather.hpp>
#include <mgpu/transfer/broadcast.hpp>

#include <test_types.hpp>
#include <sequences.hpp>
#include <helper_functions.hpp>

using namespace mgpu;
using namespace mgpu::unit;
namespace ublas = boost::numeric::ublas;

BOOST_GLOBAL_FIXTURE(environment);

// helper struct
template <typename T, std::size_t N>
struct blas_backend_data
{
    blas_backend_data(std::size_t devices) :
      X_(devices*N), Y_(devices*N), r_(devices), rbuff_(devices),
      X(devices*N), Y(devices*N), r(devices),
      r__(0)
    {
      std::generate(X_.begin(), X_.end(), random_sequence<T>());
      std::generate(Y_.begin(), Y_.end(), random_sequence<T>());
    };

    ublas::vector<T> X_;
    ublas::vector<T> Y_;
    ublas::vector<T> r_;
    ublas::vector<T> rbuff_;

    seg_dev_vector<T> X;
    seg_dev_vector<T> Y;
    seg_dev_vector<T> r;

    // gold results
    T r__;
};


// inner_prod
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(inner_prod, Type, numeric_test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  blas_backend_data<T, num> d(devices);
  blas b;

  // upload X and y
  scatter(d.X_, d.X.begin());
  scatter(d.Y_, d.Y.begin());

  // do calculation
  b.set_scalar_host();
  b.inner_prod(d.X, d.Y, d.r_.begin());

  b.set_scalar_device();
  b.inner_prod(d.X, d.Y, d.r.begin());

  gather(d.r, d.rbuff_.begin());

  // calculate gold value
  d.r__ = ublas::inner_prod(d.X_, d.Y_);

  synchronize_barrier();
//  // test host and device result
  BOOST_CHECK_CLOSE(std::abs(d.r__),
    std::abs(std::accumulate(d.r_.begin(), d.r_.end(), T(0))), 0.00001);
  BOOST_CHECK_CLOSE(std::abs(d.r__),
    std::abs(std::accumulate(d.rbuff_.begin(), d.rbuff_.end(), T(0))), 0.00001);
}
