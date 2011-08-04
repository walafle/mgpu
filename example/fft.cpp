// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#include <vector>

#include <mgpu/fft.hpp>
#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/transfer/copy.hpp>
#include <mgpu/synchronization.hpp>

using namespace mgpu;

int main(void)
{
  environment e;
  {
    unsigned int dim = 128;
    unsigned int batch = 15;

    std::size_t blocksize = dim*dim;
    std::size_t size = blocksize*batch;

    std::vector<std::complex<float> > host_in(size, std::complex<float>(0));
    std::vector<std::complex<float> > host_out(size, std::complex<float>(0));

    std::generate(host_in.begin(), host_in.end(), rand);

    seg_dev_vector<std::complex<float> > in(size, blocksize);
    seg_dev_vector<std::complex<float> > out(size, blocksize);

    copy(host_in, in.begin());

    // plan 2D FFT batch with dimension and batch
    fft<std::complex<float>, std::complex<float> > f(dim, dim, batch);

    f.forward(in, out);
    f.inverse(out, in);

    // fetch result
    copy(in, host_out.begin());
    synchronize_barrier();
  }
}
