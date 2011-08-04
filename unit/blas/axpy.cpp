// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE blas.seg_axpy

#include <math.h>

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
      X_(devices*N), Y_(devices*N), alpha_(1),
      X(devices*N), Y(devices*N), alpha(devices),
      Y__(devices*N)
    {
      std::generate(X_.begin(), X_.end(), random_sequence<T>());
      std::generate(Y_.begin(), Y_.end(), random_sequence<T>());
      alpha_[0] = random_sequence<T>()();
    };

    ublas::vector<T> X_;
    ublas::vector<T> Y_;
    ublas::vector<T> alpha_;

    seg_dev_vector<T> X;
    seg_dev_vector<T> Y;
    seg_dev_vector<T> alpha;

    // gold results
    ublas::vector<T> Y__;
};


// axpy
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(axpy, Type, numeric_test_types_and_sizes)
{
  const int num = Type::first_type::value;
  typedef typename Type::second_type T;

  int devices = environment::get_size();

  blas_backend_data<T, num> d(devices);
  blas b;

  d.Y__ = d.Y_;

  seg_dev_vector<T> Y1(num*devices);
  seg_dev_vector<T> Y2(num*devices);

  // broadcast alpha to device
  broadcast(d.alpha_, d.alpha.begin());

  // scatter X and Y to device
  scatter(d.X_, d.X.begin());
  scatter(d.Y_, Y1.begin());
  scatter(d.Y_, Y2.begin());

  // calculate axpy
  b.set_scalar_device();
  b.axpy(d.alpha.begin(), d.X, Y1);

  b.set_scalar_host();
  b.axpy(&(d.alpha_[0]), d.X, Y2);

  // calculate gold result
  d.Y__ += d.alpha_[0] * d.X_;

  // gather results from device and compare
  gather(Y1, d.Y_.begin());
  synchronize_barrier();
  for(int i=0; i<num*devices; ++i)
  {
    BOOST_CHECK_CLOSE(std::abs(d.Y_[i]), std::abs(d.Y__[i]), 0.00001);
  }

  gather(Y2, d.Y_.begin());
  synchronize_barrier();
  for(int i=0; i<num*devices; ++i)
  {
    BOOST_CHECK_CLOSE(std::abs(d.Y_[i]), std::abs(d.Y__[i]), 0.00001);
  }

}
