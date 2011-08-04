// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE fft.fft

#include <boost/test/unit_test.hpp>

#include <mgpu/fft.hpp>
#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/transfer/scatter.hpp>
#include <mgpu/transfer/gather.hpp>
#include <mgpu/synchronization.hpp>

#include <test_types.hpp>

using namespace mgpu;
using namespace mgpu::unit;

BOOST_GLOBAL_FIXTURE(environment);

// calculation_2d
// _____________________________________________________________________________

BOOST_AUTO_TEST_CASE_TEMPLATE(calculation_2d, T, fft_test_types)
{
  typedef typename boost::mpl::at_c<T, 0>::type T1;
  typedef typename boost::mpl::at_c<T, 1>::type T2;

  unsigned int dim = 128;
  unsigned int batch = 15;
  std::size_t size = dim*dim*batch;
  std::size_t blocksize = dim*dim;


  std::vector<std::complex<float> > host_in(size, std::complex<float>(0));
  std::vector<std::complex<float> > host_out(size, std::complex<float>(0));

  std::generate(host_in.begin(), host_in.end(), rand);

  seg_dev_vector<std::complex<float> > in(size, blocksize);
  seg_dev_vector<std::complex<float> > out(size, blocksize);

  scatter(host_in, in.begin());

  fft<std::complex<float>, std::complex<float> > f(dim, dim, 15);

  f.forward(in, out);
  f.inverse(out, in);

  gather(in, host_out.begin());
  synchronize_barrier();

}
