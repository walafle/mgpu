// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



/**
 * Reduce micro-benchmark
 */

//#include <stdio.h>
#include <iostream>
#include <vector>
#include <complex>

#include <boost/iterator/iterator_facade.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <stopwatch.hpp>

#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/environment.hpp>
#include <mgpu/synchronization.hpp>
#include <mgpu/invoke.hpp>
#include <mgpu/transfer/copy.hpp>
#include <mgpu/backend/dev_management.hpp>

using namespace mgpu;


// config -----


typedef double T;
#define CONSTRUCT_T T(.42)
/*
typedef float2 T;
#define CONSTRUCT_T make_float2(.42, .42)
bool operator ==(const T& a, const T& b)
{
  return(a.x==b.x && a.y==b.y);
}


typedef float4 T;
#define CONSTRUCT_T make_float4(.42, .42, .42, .42)
bool operator ==(const T& a, const T& b)
{
  return(a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w);
}
*/


int iterations = 500;
unsigned int size = 1024*1024;
unsigned int blocks = 1024;
unsigned int threads = 256;


// constant iterator for simple comparison -----

template <typename U>
struct const_iterator
  : public boost::iterator_facade< const_iterator<U>, U,
    boost::random_access_traversal_tag>
{
    const_iterator(U * value) : value_(value) {}
    void increment() { }
    bool equal(const_iterator<U> const& other) const
    { return *(this->value_) == *(other.value_); }
    U & dereference() const { return *value_; }
    friend class boost::iterator_core_access;
    U * value_;
};


// simple copy -----

__global__ void simple_copy_kernel(T * dst, T const * src,
  unsigned int const len)
{
  int stride = gridDim.x * blockDim.x;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = tid; i < len; i += stride)
  {
      dst[i] = src[i];
  }
}

void simple_copy(dev_vector<T> const & in, dev_vector<T> & out)
{
  // warm up
  simple_copy_kernel<<<blocks, threads>>>(
    out.get_raw_pointer(), in.get_raw_pointer(), in.size());

  // benchmark
  backend::sync_dev();
  sw_start("simple_copy");
  for(int i=0; i<iterations; i++)
  {
    simple_copy_kernel<<<blocks, threads>>>(
      out.get_raw_pointer(), in.get_raw_pointer(), in.size());
  }
  backend::sync_dev();
  sw_stop("simple_copy");

  // check if result is ok
  std::vector<T> out_host_add(size/sizeof(T));
  copy(out, out_host_add.begin());

  T add_result = CONSTRUCT_T;
  const_iterator<T> add_result_it(&add_result);

  if(!std::equal(out_host_add.begin(), out_host_add.end(), add_result_it))
    printf("simple_copy NOT ok\n");
}


// unrolled copy -----

template <int N>
__global__ void unrolled_copy_kernel(T * dst, T const * src,
  unsigned int const size)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

//  if(i >= size) return;
//
//  // only copy data
//  if(N == 1)
//  {
//    dst[i] = src[i];
//    return;
//  }

  unsigned long int s = 0;
#pragma unroll
  for(int j=0; j<N; j++)
  {
    dst[s+i] = src[s+i];
    s += size;
  }
}

#define UNROLLED_COPY_KEREL(z, N, text)                                        \
    case N:                                                                    \
      unrolled_copy_kernel<N><<<blocks, threads>>>(                            \
        out.get_raw_pointer(), in.get_raw_pointer(), active_threads);          \
      break;                                                                   \
    /**/


void unrolled_copy(dev_vector<T> const & in, dev_vector<T> & out)
{
  int active_threads = threads * blocks;

  std::size_t n = in.size() / active_threads;

  if(in.size() % active_threads != 0)
  {
    printf("size currently not supported\n");
    return;
  }

  // warm up
  switch (n)
  {
    BOOST_PP_REPEAT_FROM_TO(1, 256, UNROLLED_COPY_KEREL, _)
    default:
      break;
  }

  // benchmark
  backend::sync_dev();
  sw_start("unrolled_copy");

  for(int i=0; i<iterations; i++)
  {
    switch (n)
    {
      BOOST_PP_REPEAT_FROM_TO(1, 256, UNROLLED_COPY_KEREL, _)
      default:
        break;
    }
  }

  backend::sync_dev();
  sw_stop("unrolled_copy");

  // check if result is ok
  std::vector<T> out_host_add(size/sizeof(T));
  copy(out, out_host_add.begin());

  T add_result = CONSTRUCT_T;
  const_iterator<T> add_result_it(&add_result);

  if(!std::equal(out_host_add.begin(), out_host_add.end(), add_result_it))
    printf("unrolled_copy NOT ok\n");
}

int main(int argc, char* argv[])
{

  int device = 0;
  if(argc >= 2)
  {
    device = atoi(argv[1]);
  }

  if(argc >= 3)
  {
    iterations = atoi(argv[2]);
  }

  if(argc >= 4)
  {
    size = atoi(argv[3]);
  }

  if(argc >= 5)
  {
    blocks = atoi(argv[4]);
  }

  if(argc >= 6)
  {
    threads = atoi(argv[5]);
  }

  if(argc >= 7)
  {
    printf("usage %s <device> <iterations> <size> <blocks> <threads>\n",
      argv[0]);
    return 0;
  }

  double amount_of_data =
    2.*((double)size*(double)iterations)/1000./1000./1000.;

  printf("iterations: %d, size: %d, blocks: %d, threads: %d, access: %fGB\n",
    iterations, size, blocks, threads, amount_of_data);

  std::vector<T> in_host(size/sizeof(T));
  in_host.assign(in_host.size(), CONSTRUCT_T);

  backend::set_dev(device);
  {
    dev_vector<T> in(size/sizeof(T));
    dev_vector<T> out(size/sizeof(T));
    copy(in_host, in.begin());

    simple_copy(in, out);
    unrolled_copy(in, out);
  }

  double ns_to_s = 1000.*1000.*1000.;

  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);

  double peak =
    (properties.memoryClockRate*1000.*(properties.memoryBusWidth/8.)*2.)/1.e9;
  double simple = amount_of_data/(sw_get_time("simple_copy")/ns_to_s);
  double unrolled = amount_of_data/(sw_get_time("unrolled_copy")/ns_to_s);

  printf("peak %.2fGB/sec | simple %.2fGB/sec (%.2f%%), "
    "unrolled %.2fGB/sec (%.2f%%)\n",
    peak, simple, 100.*simple/peak, unrolled, 100.*unrolled/peak);

  return 0;
}
