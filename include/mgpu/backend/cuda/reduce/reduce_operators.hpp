// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_REDUCE_REDUCE_OPERATORS_HPP
#define MGPU_BACKEND_CUDA_REDUCE_REDUCE_OPERATORS_HPP

/**
 * @file reduce_operators.hpp
 *
 * This header provides the operators that can be used in reduce functions
 */

#include <mgpu/backend/cuda/cuda_types.hpp>

namespace mgpu
{

namespace backend_detail
{

namespace cuda
{

/// add reduce operator
template <typename T>
struct reduce_operator_add
{
#ifdef __CUDACC__
  typedef typename cuda_type<T>::type type;

  __host__ __device__ __forceinline__
  type operator()(type const & x, type const & y)
  {
    return x + y;
  }
#else
  typedef T type;

  inline type operator()(type const & x, type const & y)
  {
    return x + y;
  }
#endif
};

/// mul reduce operator
template <typename T>
struct reduce_operator_mul
{
#ifdef __CUDACC__
  typedef typename cuda_type<T>::type type;

  __host__ __device__ __forceinline__
  type operator()(type const & x, type const & y)
  {
    return x * y;
  }
#else
  typedef T type;

  inline type operator()(type const & x, type const & y)
  {
    return x * y;
  }
#endif
};

/// Boost preprocessor sequence of operators to generate all required overloads
#define MGPU_CUDA_DEVICE_REDUCE_OPERATORS                                      \
  (reduce_operator_add)(reduce_operator_mul)

} // namespace cuda

} // namespace backend_detail

} // namespace mgpu

#endif // MGPU_BACKEND_CUDA_REDUCE_REDUCE_OPERATORS_HPP

