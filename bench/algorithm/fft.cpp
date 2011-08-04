// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



/**
 * This fft benchmark compares FFT runtimes.
 */

#include <stdio.h>
#include <iostream>
#include <vector>
#include <complex>

#include <stopwatch.hpp>

#include <mgpu/environment.hpp>
#include <mgpu/synchronization.hpp>
#include <mgpu/backend/backend.hpp>
#include <mgpu/fft.hpp>
#include <mgpu/transfer/copy.hpp>

using namespace mgpu;

// config -----

int devices = 1;
int iterations = 500;
unsigned int dim = 256;
unsigned int batch = 16;


// input data -----

std::vector<std::complex<float> > inputdata;

// -----

std::vector<std::complex<float> > bench_mgpu_backend()
{
  std::vector<std::complex<float> > result(dim*dim*batch);
  dev_vector<std::complex<float> > A(dim*dim*batch);
  dev_vector<std::complex<float> > B(dim*dim*batch);

  copy(inputdata, A.begin());

  backend::fft<std::complex<float>, std::complex<float> > plan(dim, dim, batch);

  plan.forward(A, B);
  plan.inverse(A, B);

  copy(B, result.begin());

  backend::sync_dev();
  sw_start("mgpu_backend");

  for(int i=0; i<iterations; i++)
  {
    plan.forward(A, B);
    plan.inverse(A, B);
  }

  backend::sync_dev();
  sw_stop("mgpu_backend");
  return result;
}

std::vector<std::complex<float> > bench_mgpu()
{
  std::vector<std::complex<float> > result(dim*dim*batch);
  environment e(dev_group::from_to(0, devices));

  {
    seg_dev_vector<std::complex<float> > A(dim*dim*batch, dim*dim);
    seg_dev_vector<std::complex<float> > B(dim*dim*batch, dim*dim);

    copy(inputdata, A.begin());

    fft<std::complex<float>, std::complex<float> > plan(dim, dim, batch);

    plan.forward(A, B);
    plan.inverse(A, B);

    copy(B, result.begin());

    synchronize_barrier();
    sw_start("mgpu");

    for(int i=0; i<iterations; i++)
    {
      plan.forward(A, B);
      plan.inverse(A, B);
    }

    synchronize_barrier();
    sw_stop("mgpu");
  }
  return result;
}

std::vector<std::complex<float> > bench_cuda_single()
{
  std::vector<std::complex<float> > result(dim*dim*batch);
  std::complex<float> * A;
  std::complex<float> * B;

  cudaMalloc(&A, sizeof(std::complex<float>)*dim*dim*batch);
  cudaMalloc(&B, sizeof(std::complex<float>)*dim*dim*batch);

  cudaMemcpy(A, &inputdata[0],
    sizeof(std::complex<float>)*inputdata.size(), cudaMemcpyHostToDevice);

  cufftHandle plan;
  int dims[2] = { dim, dim };
  int embed[2] = { dim * dim, dim };
  cufftPlanMany(&plan, 2, dims, embed, 1, dim * dim, embed, 1,
    dim * dim, CUFFT_C2C, batch);

  cufftExecC2C(plan, (cufftComplex *)A, (cufftComplex *)B, CUFFT_FORWARD);
  cufftExecC2C(plan, (cufftComplex *)A, (cufftComplex *)B, CUFFT_INVERSE);

  cudaDeviceSynchronize();
  cudaMemcpy(&result[0], B,
    sizeof(std::complex<float>)*result.size(), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  sw_start("cuda_single");

  for(int i=0; i<iterations; i++)
  {
    cufftExecC2C(plan, (cufftComplex *)A, (cufftComplex *)B, CUFFT_FORWARD);
    cufftExecC2C(plan, (cufftComplex *)A, (cufftComplex *)B, CUFFT_INVERSE);
  }

  cudaDeviceSynchronize();
  sw_stop("cuda_single");

  cufftDestroy(plan);
  cudaFree(B);
  cudaFree(A);
  return result;
}

std::vector<std::complex<float> > bench_cuda_multi()
{
  std::vector<std::complex<float> > result(dim*dim*batch);

  // calculate batch per device
  int batch_per_device = batch / devices;
  int batch_rest = batch % devices;
  boost::array<int, MGPU_NUM_DEVICES> bpd;
  for(int i=0; i<devices; i++)
  {
    bpd[i] = batch_per_device + ((batch_rest>0) ? 1 : 0);
    batch_rest--;
  }

  boost::array<std::complex<float> *, MGPU_NUM_DEVICES> A;
  boost::array<std::complex<float> *, MGPU_NUM_DEVICES> B;
  boost::array<cufftHandle, MGPU_NUM_DEVICES> plan;
  boost::array<cudaStream_t, MGPU_NUM_DEVICES> stream;

  int running_bpd = 0;
  for(int i=0; i<devices; i++)
  {
    cudaSetDevice(i);
    cudaMalloc(&A[i], sizeof(std::complex<float>)*dim*dim*bpd[i]);
    cudaMalloc(&B[i], sizeof(std::complex<float>)*dim*dim*bpd[i]);

    cudaMemcpy(A[i], &inputdata[dim*dim*running_bpd],
      sizeof(std::complex<float>)*dim*dim*bpd[i], cudaMemcpyHostToDevice);

    int dims[2] = { dim, dim };
    int embed[2] = { dim * dim, dim };
    cufftPlanMany(&plan[i], 2, dims, embed, 1, dim * dim, embed, 1,
      dim * dim, CUFFT_C2C, bpd[i]);
    cudaStreamCreate(&stream[i]);
    cufftSetStream(plan[i], stream[i]);
    running_bpd += bpd[i];
  }

  for(int i=0; i<devices; i++)
  {
    cudaSetDevice(i);
    cufftExecC2C(plan[i],
      (cufftComplex *)A[i], (cufftComplex *)B[i], CUFFT_FORWARD);
    cufftExecC2C(plan[i],
      (cufftComplex *)A[i], (cufftComplex *)B[i], CUFFT_INVERSE);
  }

  running_bpd = 0;
  for(int i=0; i<devices; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
    cudaMemcpy(&result[dim*dim*running_bpd], B[i],
      sizeof(std::complex<float>)*dim*dim*bpd[i], cudaMemcpyDeviceToHost);
    running_bpd += bpd[i];
  }

  for(int i=0; i<devices; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
  sw_start("cuda_multi");

  for(int i=0; i<iterations; i++)
  {
    for(int i=0; i<devices; i++)
    {
      sw_start("cuda_multi_inner");
      cudaSetDevice(i);
      cufftExecC2C(plan[i],
        (cufftComplex *)A[i], (cufftComplex *)B[i], CUFFT_FORWARD);
      cufftExecC2C(plan[i],
        (cufftComplex *)A[i], (cufftComplex *)B[i], CUFFT_INVERSE);
      sw_stop("cuda_multi_inner");
    }
  }

  for(int i=0; i<devices; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
  sw_stop("cuda_multi");

  for(int i=0; i<devices; i++)
  {
    cudaSetDevice(i);
    cufftDestroy(plan[i]);
    cudaFree(B[i]);
    cudaFree(A[i]);
    cudaStreamDestroy(stream[i]);
  }
  return result;
}


int main(int argc, char* argv[])
{

  if(argc >= 2)
  {
    if(atoi(argv[1]) > MGPU_NUM_DEVICES)
    {
      printf("only %d devices available\n", devices);
      return 0;
    }
    devices = atoi(argv[1]);
  }

  if(argc >= 3)
  {
    iterations = atoi(argv[2]);
  }

  if(argc >= 4)
  {
    dim = atoi(argv[3]);
  }

  if(argc >= 5)
  {
    batch = atoi(argv[4]);
  }


  if(argc >= 6)
  {
    printf("usage %s <devices> <iterations> <dimension> <batch>\n", argv[0]);
    return 0;
  }


  {
    printf("%d devices, %d iterations: %d x %dx%d\n",
      devices, iterations, batch, dim, dim);

    // generate input data
    inputdata.resize(dim*dim*batch);
    inputdata.assign(inputdata.size(), std::complex<float>(.1, .2));

    // run tests
    std::vector<std::complex<float> > r1 = bench_mgpu_backend();
    std::vector<std::complex<float> > r2 = bench_mgpu();
    std::vector<std::complex<float> > r3 = bench_cuda_single();
    std::vector<std::complex<float> > r4 = bench_cuda_multi();

    // compare result
    if(std::equal(r1.begin(), r1.end(), r2.begin()) == true &&
      std::equal(r1.begin(), r1.end(), r3.begin()) == true &&
      std::equal(r1.begin(), r1.end(), r4.begin()) == true)
    {
      printf("test ok\n");
    }
    else
    {
      printf("test not ok\n");
      return 0;
    }
  }
  sw_print2();
  return 0;
}
