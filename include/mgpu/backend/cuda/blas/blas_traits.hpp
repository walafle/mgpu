// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_BLAS_BLAS_TRAITS_HPP
#define MGPU_BACKEND_CUDA_BLAS_BLAS_TRAITS_HPP

/**
 * @file blas_traits.hpp
 *
 * This header provides BLAS specific type traits
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/vector_c.hpp>

#include <mgpu/core/dev_ptr.hpp>
#include <mgpu/container/range_traits.hpp>
#include <mgpu/backend/cuda/blas/exception.hpp>
#include <mgpu/backend/cuda/cuda_types.hpp>


namespace mgpu
{

namespace backend_detail
{

namespace cuda
{




// how do 2 types relate to each other in terms of their size -----

template <typename From, typename To>
struct cuda_type_relation
{
  static inline std::size_t size(std::size_t const & s) { return s; }
};

template <typename T>
struct cuda_type_relation<T, std::complex<T> >
{
  static inline std::size_t size(std::size_t const & s) { return s/2; }
};

template <typename T>
struct cuda_type_relation<std::complex<T>, T>
{
  static inline std::size_t size(std::size_t const & s) { return s*2; }
};


#define MGPU_CUDA_BLAS_GEN_INNER_PROD(functionname, cudafunctionname)          \
  template <typename ResultIterator>                                           \
  static void functionname(cublasHandle_t & handle                             \
    , const dev_ptr<RangeType> & x, const dev_ptr<RangeType> & y               \
    , ResultIterator & result                                                  \
    , const std::size_t n, const std::size_t incx = 1,                         \
    const std::size_t incy = 1)                                                \
  {                                                                            \
    MGPU_CUDA_BLAS_CALL(cudafunctionname(handle                                \
      , cuda_type_relation<RangeType, ScalarType>::size(n)                     \
      , reinterpret_cast<const typename cuda_type<ScalarType>::type *>(        \
          x.get_raw_pointer())                                                 \
      , incx                                                                   \
      , reinterpret_cast<const typename cuda_type<ScalarType>::type *>(        \
          y.get_raw_pointer())                                                 \
      , incy                                                                   \
      , reinterpret_cast<typename cuda_type<ScalarType>::type *>(              \
          ::mgpu::iterator_traits<ResultIterator>::get_raw_pointer(result)     \
      )                                                                        \
    ));                                                                        \
  }                                                                            \
  /**/

#define MGPU_CUDA_BLAS_GEN_AXPY(functionname, cudafunctionname)                \
  template <typename AlphaIterator>                                            \
  static void functionname(cublasHandle_t handle, AlphaIterator & alpha        \
    , const dev_ptr<RangeType> x, dev_ptr<RangeType> y                         \
    , const std::size_t n, const std::size_t incx = 1                          \
    , const std::size_t incy = 1)                                              \
  {                                                                            \
    MGPU_CUDA_BLAS_CALL(cudafunctionname(handle                                \
      , cuda_type_relation<RangeType, ScalarType>::size(n)                     \
      , reinterpret_cast<const typename cuda_type<ScalarType>::type *>(        \
          ::mgpu::iterator_traits<AlphaIterator>::get_raw_pointer(alpha))      \
      , reinterpret_cast<const typename cuda_type<ScalarType>::type *>(        \
          x.get_raw_pointer())                                                 \
      , incx                                                                   \
      , reinterpret_cast<typename cuda_type<ScalarType>::type *>(              \
          y.get_raw_pointer()), incy                                           \
      )                                                                        \
    );                                                                         \
  }                                                                            \
  /**/


// map types to blas methods -----

template <typename RangeType, typename ScalarType>
struct blas_traits : boost::mpl::int_<-1>
{
  BOOST_MPL_ASSERT_MSG(
      false
    , PLATFORM_CUDA_BLAS_TYPES_NOT_SUPPORTED
    , (RangeType, ScalarType)
    );
};

template <>
struct blas_traits<float, float>
{
  typedef float RangeType;
  typedef float ScalarType;

  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod, cublasSdot)

  MGPU_CUDA_BLAS_GEN_AXPY(axpy, cublasSaxpy)
};

template <>
struct blas_traits<std::complex<float>, std::complex<float> >
{
  typedef std::complex<float> RangeType;
  typedef std::complex<float> ScalarType;

  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod, cublasCdotu)
  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod_c, cublasCdotc)
  MGPU_CUDA_BLAS_GEN_AXPY(axpy, cublasCaxpy)
};

template <>
struct blas_traits<float, std::complex<float> >
{
  typedef float RangeType;
  typedef std::complex<float> ScalarType;

  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod, cublasCdotu)
  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod_c, cublasCdotc)
  MGPU_CUDA_BLAS_GEN_AXPY(axpy, cublasCaxpy)
};

template <>
struct blas_traits<std::complex<float>, float>
{
  typedef std::complex<float> RangeType;
  typedef float ScalarType;

  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod, cublasSdot)
  MGPU_CUDA_BLAS_GEN_AXPY(axpy, cublasSaxpy)
};

// double types

template <>
struct blas_traits<double, double>
{
  typedef double RangeType;
  typedef double ScalarType;

  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod, cublasDdot)
  MGPU_CUDA_BLAS_GEN_AXPY(axpy, cublasDaxpy)
};

template <>
struct blas_traits<std::complex<double>, std::complex<double> >
{
  typedef std::complex<double> RangeType;
  typedef std::complex<double> ScalarType;

  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod, cublasZdotu)
  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod_c, cublasZdotc)
  MGPU_CUDA_BLAS_GEN_AXPY(axpy, cublasZaxpy)
};

template <>
struct blas_traits<double, std::complex<double> >
{
  typedef double RangeType;
  typedef std::complex<double> ScalarType;

  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod, cublasZdotu)
  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod_c, cublasZdotc)
  MGPU_CUDA_BLAS_GEN_AXPY(axpy, cublasZaxpy)
};

template <>
struct blas_traits<std::complex<double>, double>
{
  typedef std::complex<double> RangeType;
  typedef double ScalarType;

  MGPU_CUDA_BLAS_GEN_INNER_PROD(inner_prod, cublasDdot)
  MGPU_CUDA_BLAS_GEN_AXPY(axpy, cublasDaxpy)
};

} // namespace cuda

} // namespace backend_detail

} // namespace mgpu

#endif // MGPU_BACKEND_CUDA_BLAS_BLAS_TRAITS_HPP
