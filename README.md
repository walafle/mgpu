MGPU
====

The MGPU library strives to simplify the implementation of high performance 
applications and algorithms on multi-GPU systems. Its main goal is both to 
abstract platform dependent functions and vendor specific APIs, as well as 
simplifying communication between different compute elements. The library is 
currently an alpha release containing only limited yet already useful 
functionality. The documentation is available 
[here](http://sschaetz.github.com/mgpu/).


News
----

* 2011-08-04 MGPU v0.1 has been released, download
  [bz2](https://github.com/sschaetz/mgpu/raw/archives/mgpu_0_1.tar.bz2) or
  [zip](https://github.com/sschaetz/mgpu/raw/archives/mgpu_0_1.zip) archives
  or fork the code from github

Examples
--------

The following example shows how a batched FFT can be calculated in parallel on 
all available GPUs in a system. 

```cpp
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
```


This example shows how a vector can be distributed across all devices and how
a kernel can be invoked to operate on the local data.



```cpp
#include <stdlib.h>
#include <algorithm>
#include <vector>

#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/transfer/copy.hpp>
#include <mgpu/invoke_kernel.hpp>
#include <mgpu/synchronization.hpp>

using namespace mgpu;

// generate random number
float random_number() { return ((float)(rand()%100) / 100); }

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
  int threads = 256;
  int blocks = (X.size() + T - 1) / T;
  axpy_kernel<<< blocks, threads >>>(a, X.get_raw_pointer(), Y.get_raw_pointer(), Y.size());
}

int main(void)
{
  const std::size_t size = 1024;
  environment e;
  {
    std::vector<float> X(size), Y(size);
    float const a = .42;
    std::generate(X.begin(), X.end(), random_number);
    std::generate(Y.begin(), Y.end(), random_number);

    seg_dev_vector<float> X_dev(size), Y_dev(size);
    copy(X, X_dev.begin()); copy(Y, Y_dev.begin());

    // calculate on devices
    invoke_kernel_all(axpy, a, X_dev, Y_dev);
    copy(Y_dev, Y.begin());
    synchronize_barrier();
    // result is now in Y
  }
}
```


Please refer to the [documentation](http://sschaetz.github.com/mgpu/>)
for further examples and for information on how to get started.

