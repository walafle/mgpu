// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_FFT_FFT_TRAITS_HPP
#define MGPU_BACKEND_CUDA_FFT_FFT_TRAITS_HPP

/**
 * @file fft_traits.hpp
 *
 * This header provides FFT specific type traits
 */

#include <cuda_runtime.h>
#include <cufft.h>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/vector_c.hpp>

#include <mgpu/core/dev_ptr.hpp>
#include <mgpu/backend/cuda/fft/exception.hpp>
#include <mgpu/backend/cuda/cuda_types.hpp>

namespace mgpu
{

namespace backend_detail
{

namespace cuda
{


// map types to fft methods -----

template <typename T, typename U>
struct fft_traits
{
  typedef boost::mpl::false_ forward_possible;
  typedef boost::mpl::false_ inverse_possible;

  BOOST_MPL_ASSERT_MSG(
      false
    , PLATFORM_CUDA_CUFFT_TYPES_NOT_SUPPORTED
    , (T, U)
    );
};

// real types
template <>
struct fft_traits<std::complex<float>, std::complex<float> >
{
  typedef std::complex<float> T;
  typedef std::complex<float> U;

  typedef boost::mpl::int_<CUFFT_C2C> id;

  typedef boost::mpl::true_ forward_possible;
  typedef boost::mpl::true_ inverse_possible;

  static inline cufftResult forward(cufftHandle & p, dev_ptr<T> i, dev_ptr<U> o)
  {
    return cufftExecC2C(p,
      (cuda_type<T>::type *)i.get_raw_pointer(),
      (cuda_type<U>::type *)o.get_raw_pointer(),
      CUFFT_FORWARD);
  }

  static inline cufftResult inverse(cufftHandle & p ,dev_ptr<T> i, dev_ptr<U> o)
  {
    return cufftExecC2C(p,
      (cuda_type<T>::type *)i.get_raw_pointer(),
      (cuda_type<U>::type *)o.get_raw_pointer(),
      CUFFT_INVERSE);
  }
};

template <>
struct fft_traits<float, std::complex<float> >
{
  typedef float T;
  typedef std::complex<float> U;

  typedef boost::mpl::int_<CUFFT_R2C> id;

  typedef boost::mpl::true_ forward_possible;
  typedef boost::mpl::false_ inverse_possible;


  static inline cufftResult forward(cufftHandle & p, dev_ptr<T> i, dev_ptr<U> o)
  {
    return cufftExecR2C(p,
      (cuda_type<T>::type *)i.get_raw_pointer(),
      (cuda_type<U>::type *)o.get_raw_pointer());
  }

  static inline cufftResult inverse(cufftHandle & p ,dev_ptr<T> i, dev_ptr<U> o)
  { return CUFFT_EXEC_FAILED; }
};

template <>
struct fft_traits<std::complex<float>, float>
{
  typedef std::complex<float> T;
  typedef float U;

  typedef boost::mpl::int_<CUFFT_C2R> id;

  typedef boost::mpl::false_ forward_possible;
  typedef boost::mpl::true_ inverse_possible;

  static inline cufftResult forward(cufftHandle & p, dev_ptr<T> i, dev_ptr<U> o)
  { return CUFFT_EXEC_FAILED; }

  static inline cufftResult inverse(cufftHandle & p ,dev_ptr<T> i, dev_ptr<U> o)
  {
    return cufftExecC2R(p,
      (cuda_type<T>::type *)i.get_raw_pointer(),
      (cuda_type<U>::type *)o.get_raw_pointer());
  }
};

// double types
template <>
struct fft_traits<std::complex<double>, std::complex<double> >
{
  typedef std::complex<double> T;
  typedef std::complex<double> U;

  typedef boost::mpl::int_<CUFFT_Z2Z> id;

  typedef boost::mpl::true_ forward_possible;
  typedef boost::mpl::true_ inverse_possible;

  static inline cufftResult forward(cufftHandle & plan,
    dev_ptr<T> in, dev_ptr<U> out)
  {
    return cufftExecZ2Z(plan,
      (cufftDoubleComplex *)in.get_raw_pointer(),
      (cufftDoubleComplex *)out.get_raw_pointer(), CUFFT_FORWARD);
  }

  static inline cufftResult inverse(cufftHandle & plan,
    dev_ptr<T> in, dev_ptr<U> out)
  {
    return cufftExecZ2Z(plan,
      (cufftDoubleComplex *)in.get_raw_pointer(),
      (cufftDoubleComplex *)out.get_raw_pointer(), CUFFT_INVERSE);
  }
};

template <>
struct fft_traits<double, std::complex<double> >
{
  typedef double T;
  typedef std::complex<double> U;

  typedef boost::mpl::int_<CUFFT_D2Z> id;

  typedef boost::mpl::true_ forward_possible;
  typedef boost::mpl::false_ inverse_possible;

  static inline cufftResult forward(cufftHandle & p, dev_ptr<T> i, dev_ptr<U> o)
  {
    return cufftExecD2Z(p,
      (cuda_type<T>::type *)i.get_raw_pointer(),
      (cuda_type<U>::type *)o.get_raw_pointer());
  }

  static inline cufftResult inverse(cufftHandle & p ,dev_ptr<T> i, dev_ptr<U> o)
  { return CUFFT_EXEC_FAILED; }
};

template <>
struct fft_traits<std::complex<double>, double>
{
  typedef std::complex<double> T;
  typedef double U;

  typedef boost::mpl::int_<CUFFT_Z2D> id;

  typedef boost::mpl::false_ forward_possible;
  typedef boost::mpl::true_ inverse_possible;

  static inline cufftResult forward(cufftHandle & p, dev_ptr<T> i, dev_ptr<U> o)
  { return CUFFT_EXEC_FAILED; }

  static inline cufftResult inverse(cufftHandle & p ,dev_ptr<T> i, dev_ptr<U> o)
  {
    return cufftExecZ2D(p,
      (cuda_type<T>::type *)i.get_raw_pointer(),
      (cuda_type<U>::type *)o.get_raw_pointer());
  }
};

} // namespace cuda

} // namespace backend_detail

} // namespace mgpu

#endif // MGPU_BACKEND_CUDA_FFT_FFT_TRAITS_HPP
