// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----

#include <stdlib.h>
#include <algorithm>

#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/transfer/copy.hpp>
#include <mgpu/invoke_kernel.hpp>
#include <mgpu/synchronization.hpp>

#include <boost/numeric/ublas/vector.hpp>

using namespace mgpu;
using namespace boost::numeric;


// generate random number
float random_number() { return ((float)(rand()%100) / 100); }

// compare floating point numbers
bool rough_eq(float lhs, float rhs, float epsilon)
{ return fabs(lhs - rhs) < epsilon; }


// axpy CUDA kernel code
__global__ void axpy_kernel(
  float const a, float * X, float * Y, std::size_t size)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) Y[i] = (a * X[i]) + Y[i];
}

// axpy CUDA kernel launcher
void axpy(float const a, dev_range<float> X, dev_range<float> Y)
{
  assert(X.size() == Y.size());
  int T = 256;
  int B = (X.size() + T - 1) / T;
  axpy_kernel<<< B, T >>>(
    a, X.get_raw_pointer(), Y.get_raw_pointer(), Y.size());
}



int main(void)
{
  const std::size_t size = 1024;
  environment e;
  {
    ublas::vector<float> X(size), Y(size), Y_gold(size);
    float const a = .42;
    std::generate(X.begin(), X.end(), random_number);
    std::generate(Y.begin(), Y.end(), random_number);
    Y_gold = Y;

    seg_dev_vector<float> X_dev(size), Y_dev(size);
    copy(X, X_dev.begin()); copy(Y, Y_dev.begin());

    // calculate on devices
    invoke_kernel_all(axpy, a, X_dev, Y_dev);
    copy(Y_dev, Y.begin());
    synchronize_barrier();

    // gold result (using boost::numeric)
    Y_gold += a * X;

    // compare result
    bool equal = true;
    for(unsigned int i=0; i<Y.size(); i++)
    { equal &= rough_eq(Y_gold[i], Y[i], .0001); }
    if(equal) printf("test ok\n"); else printf("test not ok\n");
  }
}
