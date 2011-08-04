// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



/**
 * Reduce micro-benchmark
 */

#include <stdio.h>
#include <iostream>
#include <vector>
#include <complex>

#include <boost/iterator/iterator_facade.hpp>

#include <stopwatch.hpp>

#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/environment.hpp>
#include <mgpu/synchronization.hpp>
#include <mgpu/invoke.hpp>
#include <mgpu/transfer/copy.hpp>
#include <mgpu/backend/dev_management.hpp>

using namespace mgpu;


// config -----

typedef float T;
int iterations = 500;
unsigned int dim = 256;
unsigned int batch = 16;
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


// simple block reduction: add -----

__global__ void simple_block_reduction_add_kernel(float * dst,
  float const * src,
  unsigned int const size, unsigned int const n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= size) return;

  // only copy data
  if(n == 1)
  {
    dst[i] = src[i];
    return;
  }

  // initialize
  T r = src[i];

  // reduce
  for(unsigned int s=i+size; s<n*size; s+=size)
  {
    r = r + src[s];
  }
  dst[i] = r;
}

void simple_block_reduction_add(seg_dev_vector<T> const & in,
  dev_vector<T> & out)
{
  std::size_t size = out.size();
  std::size_t n = in.blocks();
  int blocks = (size + threads - 1) / threads;

  synchronize_barrier();

  // warm up
  simple_block_reduction_add_kernel<<<blocks, threads>>>(
    out.get_raw_pointer(), in.get_raw_pointer(0), size, n);

  // benchmark
  synchronize_barrier();
  sw_start("simple_block_reduction_add");
  for(int i=0; i<iterations; i++)
  {
    simple_block_reduction_add_kernel<<<blocks, threads>>>(
      out.get_raw_pointer(), in.get_raw_pointer(0), size, n);
  }
  synchronize_barrier();
  sw_stop("simple_block_reduction_add");

  // check if result is ok
  std::vector<T> out_host_add(dim*dim);
  copy(out, out_host_add.begin());

  T add_result = T(0);
  for(unsigned int i=0; i<batch; i++) add_result+=T(.42);
  const_iterator<T> add_result_it(&add_result);

  if(!std::equal(out_host_add.begin(), out_host_add.end(), add_result_it))
    printf("simple_block_reduction_add NOT ok\n");
}




// two at a time block reduction: add -----

inline __device__ float2 operator+ (float2 a, float2 b)
{ return make_float2(a.x + b.x, a.y + b.y); }

__global__ void two_at_a_time_block_reduction_add_kernel(float2 * dst,
  float2 const * src, unsigned int const size, unsigned int const n)
{
  unsigned int i = blockIdx.x * blockDim.x + (threadIdx.x);

  if(i >= size) return;

  // only copy data
  if(n == 1)
  {
    dst[i] = src[i];
    return;
  }

  // initialize
  float2 r = src[i];

  // reduce
  unsigned int s = size;
  for(unsigned int m=1; m<n; m++)
  {
    r = r + src[s + i];
    s += size;
  }
  dst[i] = r;
}

void two_at_a_time_block_reduction_add(seg_dev_vector<T> const & in,
  dev_vector<T> & out)
{
  std::size_t size = out.size() / 2;
  std::size_t n = in.blocks();
  int blocks = ((size) + threads - 1) / threads;

  synchronize_barrier();

  // warm up
  two_at_a_time_block_reduction_add_kernel<<<blocks, threads>>>(
    (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size, n);

  // benchmark
  synchronize_barrier();
  sw_start("two_at_a_time_block_reduction_add");
  for(int i=0; i<iterations; i++)
  {
    two_at_a_time_block_reduction_add_kernel<<<blocks, threads>>>(
      (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size, n);
  }
  synchronize_barrier();
  sw_stop("two_at_a_time_block_reduction_add");

  // check if result is ok
  std::vector<T> out_host_add(dim*dim);
  copy(out, out_host_add.begin());

  T add_result = T(0);
  for(unsigned int i=0; i<batch; i++) add_result+=T(.42);
  const_iterator<T> add_result_it(&add_result);

  if(!std::equal(out_host_add.begin(), out_host_add.end(), add_result_it))
    printf("two_at_a_time_block_reduction_add NOT ok\n");
}



// unrolled block reduction: add -----

template <int N>
__global__ void unrolled_block_reduction_add_kernel(float * dst,
  float const * src, unsigned int const size)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= size) return;

  // only copy data
  if(N == 1)
  {
    dst[i] = src[i];
    return;
  }

  // initialize
  T r = src[i];

  // reduce
  unsigned int s = size;
#pragma unroll
  for(unsigned int m=1; m<N; m++)
  {
    r = r + src[s + i];
    s += size;
  }
  dst[i] = r;
}

template <>
__global__ void unrolled_block_reduction_add_kernel<1>(float * dst,
  float const * src, unsigned int const size)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= size) return;

  // only copy data
  dst[i] = src[i];
  return;
}

void unrolled_block_reduction_add(seg_dev_vector<T> const & in,
  dev_vector<T> & out)
{
  std::size_t size = out.size();
  std::size_t n = in.blocks();
  int blocks = (size + threads - 1) / threads;

  synchronize_barrier();

  // warm up
  switch (n)
  {
    case 1:
      unrolled_block_reduction_add_kernel<1><<<blocks, threads>>>(
        out.get_raw_pointer(), in.get_raw_pointer(0), size);
      break;
    case 2:
      unrolled_block_reduction_add_kernel<2><<<blocks, threads>>>(
        out.get_raw_pointer(), in.get_raw_pointer(0), size);
      break;
    case 3:
      unrolled_block_reduction_add_kernel<3><<<blocks, threads>>>(
        out.get_raw_pointer(), in.get_raw_pointer(0), size);
      break;
    case 4:
      unrolled_block_reduction_add_kernel<4><<<blocks, threads>>>(
        out.get_raw_pointer(), in.get_raw_pointer(0), size);
      break;
    case 8:
      unrolled_block_reduction_add_kernel<8><<<blocks, threads>>>(
        out.get_raw_pointer(), in.get_raw_pointer(0), size);
      break;
    case 16:
      unrolled_block_reduction_add_kernel<16><<<blocks, threads>>>(
        out.get_raw_pointer(), in.get_raw_pointer(0), size);
      break;
    case 32:
      unrolled_block_reduction_add_kernel<32><<<blocks, threads>>>(
        out.get_raw_pointer(), in.get_raw_pointer(0), size);
      break;
    default:
      break;
  }

  // benchmark
  synchronize_barrier();
  sw_start("unrolled_block_reduction_add");
  for(int i=0; i<iterations; i++)
  {
    switch (n)
    {
      case 1:
        unrolled_block_reduction_add_kernel<1><<<blocks, threads>>>(
          out.get_raw_pointer(), in.get_raw_pointer(0), size);
        break;
      case 2:
        unrolled_block_reduction_add_kernel<2><<<blocks, threads>>>(
          out.get_raw_pointer(), in.get_raw_pointer(0), size);
        break;
      case 3:
        unrolled_block_reduction_add_kernel<3><<<blocks, threads>>>(
          out.get_raw_pointer(), in.get_raw_pointer(0), size);
        break;
      case 4:
        unrolled_block_reduction_add_kernel<4><<<blocks, threads>>>(
          out.get_raw_pointer(), in.get_raw_pointer(0), size);
        break;
      case 8:
        unrolled_block_reduction_add_kernel<8><<<blocks, threads>>>(
          out.get_raw_pointer(), in.get_raw_pointer(0), size);
        break;
      case 16:
        unrolled_block_reduction_add_kernel<16><<<blocks, threads>>>(
          out.get_raw_pointer(), in.get_raw_pointer(0), size);
        break;
      case 32:
        unrolled_block_reduction_add_kernel<32><<<blocks, threads>>>(
          out.get_raw_pointer(), in.get_raw_pointer(0), size);
        break;
      default:
        break;
    }
  }
  synchronize_barrier();
  sw_stop("unrolled_block_reduction_add");

  // check if result is ok
  std::vector<T> out_host_add(dim*dim);
  copy(out, out_host_add.begin());

  T add_result = T(0);
  for(unsigned int i=0; i<batch; i++) add_result+=T(.42);
  const_iterator<T> add_result_it(&add_result);

  if(!std::equal(out_host_add.begin(), out_host_add.end(), add_result_it))
    printf("unrolled_block_reduction_add NOT ok\n");
}


// unrolled block reduction two at a time: add -----

template <int N>
__global__ void unrolled_block_reduction_two_add_kernel(
  float2 * dst, float2 * src, unsigned int const size)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= size) return;

  // only copy data
  if(N == 1)
  {
    dst[i] = src[i];
    return;
  }

  // initialize
  float2 r = src[i];

  // reduce
  unsigned int s = size;
#pragma unroll
  for(unsigned int m=1; m<N; m++)
  {
    r = r + src[s + i];
    s += size;
  }
  dst[i] = r;
}

template <>
__global__ void unrolled_block_reduction_two_add_kernel<1>(
  float2 * dst, float2 * src, unsigned int const size)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= size) return;

  // only copy data
  dst[i] = src[i];
  return;
}

void unrolled_block_reduction_two_add(seg_dev_vector<T> const & in,
  dev_vector<T> & out)
{
  std::size_t size = out.size() / 2;
  std::size_t n = in.blocks();
  int blocks = (size + threads - 1) / threads;

  synchronize_barrier();

  // warm up
  switch (n)
  {
    case 1:
      unrolled_block_reduction_two_add_kernel<1><<<blocks, threads>>>(
        (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
      break;
    case 2:
      unrolled_block_reduction_two_add_kernel<2><<<blocks, threads>>>(
        (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
      break;
    case 3:
      unrolled_block_reduction_two_add_kernel<3><<<blocks, threads>>>(
        (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
      break;
    case 4:
      unrolled_block_reduction_two_add_kernel<4><<<blocks, threads>>>(
        (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
      break;
    case 8:
      unrolled_block_reduction_two_add_kernel<8><<<blocks, threads>>>(
        (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
      break;
    case 16:
      unrolled_block_reduction_two_add_kernel<16><<<blocks, threads>>>(
        (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
      break;
    case 32:
      unrolled_block_reduction_two_add_kernel<32><<<blocks, threads>>>(
        (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
      break;
    default:
      break;
  }

  // benchmark
  synchronize_barrier();
  sw_start("unrolled_block_reduction_two_add");
  for(int i=0; i<iterations; i++)
  {
    switch (n)
    {
      case 1:
        unrolled_block_reduction_two_add_kernel<1><<<blocks, threads>>>(
          (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
        break;
      case 2:
        unrolled_block_reduction_two_add_kernel<2><<<blocks, threads>>>(
          (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
        break;
      case 3:
        unrolled_block_reduction_two_add_kernel<3><<<blocks, threads>>>(
          (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
        break;
      case 4:
        unrolled_block_reduction_two_add_kernel<4><<<blocks, threads>>>(
          (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
        break;
      case 8:
        unrolled_block_reduction_two_add_kernel<8><<<blocks, threads>>>(
          (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
        break;
      case 16:
        unrolled_block_reduction_two_add_kernel<16><<<blocks, threads>>>(
          (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
        break;
      case 32:
        unrolled_block_reduction_two_add_kernel<32><<<blocks, threads>>>(
          (float2*)out.get_raw_pointer(), (float2*)in.get_raw_pointer(0), size);
        break;
      default:
        break;
    }
  }
  synchronize_barrier();
  sw_stop("unrolled_block_reduction_two_add");

  // check if result is ok
  std::vector<T> out_host_add(dim*dim);
  copy(out, out_host_add.begin());

  T add_result = T(0);
  for(unsigned int i=0; i<batch; i++) add_result+=T(.42);
  const_iterator<T> add_result_it(&add_result);

  if(!std::equal(out_host_add.begin(), out_host_add.end(), add_result_it))
    printf("unrolled_block_reduction_two_add NOT ok\n");
}


// simple template block reduction: add -----

template <typename U>
struct reduce_functor_add
{
  __host__ __device__ __forceinline__
  U operator()(U const & x, U const & y)
  {
    return x + y;
  }
};

// a template template argument
template <typename U, template <class> class Op>
__global__ void simple_template_block_reduction_add_kernel(
  U * dst, U const * src, unsigned int const size, unsigned int const n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i >= size) return;

  // only copy data
  if(n == 1)
  {
    dst[i] = src[i];
    return;
  }

  // initialize
  U r = src[i];

  // reduce
  for(unsigned int s=i+size; s<n*size; s+=size)
  {
    r = Op<U>()(r, src[s]);
  }
  dst[i] = r;
}

void simple_template_block_reduction_add(seg_dev_vector<float> const & in,
  dev_vector<float> & out)
{
  unsigned int const size = out.size();
  unsigned int const n = in.blocks();
  int blocks = (size + threads - 1) / threads;

  synchronize_barrier();

  // warm up
  simple_template_block_reduction_add_kernel<T, reduce_functor_add >
    <<<blocks, threads>>>(out.get_raw_pointer(),
      in.get_raw_pointer(0), size, n);

  // benchmark
  synchronize_barrier();
  sw_start("simple_template_block_reduction_add");
  for(int i=0; i<iterations; i++)
  {
    simple_template_block_reduction_add_kernel<float, reduce_functor_add >
      <<<blocks, threads>>>(out.get_raw_pointer(),
        in.get_raw_pointer(0), size, n);
  }
  synchronize_barrier();
  sw_stop("simple_template_block_reduction_add");

  // check if result is ok
  std::vector<T> out_host_add(dim*dim);
  copy(out, out_host_add.begin());

  T add_result = T(0);
  for(unsigned int i=0; i<batch; i++) add_result+=T(.42);
  const_iterator<T> add_result_it(&add_result);

  if(!std::equal(out_host_add.begin(), out_host_add.end(), add_result_it))
    printf("simple_template_block_reduction_add NOT ok\n");
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
    dim = atoi(argv[3]);
  }

  if(argc >= 5)
  {
    batch = atoi(argv[4]);
  }

  if(argc >= 6)
  {
    threads = atoi(argv[5]);
  }

  if(argc >= 7)
  {
    printf("usage %s <device> <iterations> <dimension> <batch> <threads>\n",
      argv[0]);
    return 0;
  }

  double amount_of_data =
    ((float)dim*dim*(batch+1)*iterations*sizeof(T))/1000/1000/1000;

  printf("%d iterations: %d x %dx%d, threads: %d access: %fGB\n",
    iterations, batch, dim, dim, threads, amount_of_data);

  std::vector<T> in_host(dim*dim*batch);
  in_host.assign(in_host.size(), T(.42));

  environment e(dev_group::from_to(device, device+1));
  synchronize_barrier();
  backend::set_dev(device);
  {
    seg_dev_vector<T> in(dim*dim*batch, dim*dim);
    dev_vector<T> out(dim*dim);
    copy(in_host, in.begin());

    simple_block_reduction_add(in, out);
    two_at_a_time_block_reduction_add(in, out);
    unrolled_block_reduction_add(in, out);
    unrolled_block_reduction_two_add(in, out);
    simple_template_block_reduction_add(in, out);
  }

  double ns_to_s = 1000.*1000.*1000.;

  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);

  double peak =
    (properties.memoryClockRate*1000.*(properties.memoryBusWidth/8.)*2.)/1.e9;



  printf("peak %.2fGB/sec | simple %.2fGB/sec, simple two %.2fGB/sec, "
    "unrolled %.2fGB/sec, unrolled two %.2fGB/sec, "
    "simple template %.2fGB/sec\n",
    peak,
    amount_of_data/(sw_get_time("simple_block_reduction_add")/ns_to_s),
    amount_of_data/(sw_get_time("two_at_a_time_block_reduction_add")/ns_to_s),
    amount_of_data/(sw_get_time("unrolled_block_reduction_add")/ns_to_s),
    amount_of_data/(sw_get_time("unrolled_block_reduction_two_add")/ns_to_s),
    amount_of_data/(sw_get_time("simple_template_block_reduction_add")/ns_to_s)
    );
  return 0;
}
