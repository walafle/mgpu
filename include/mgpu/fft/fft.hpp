// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_FFT_FFT_HPP
#define MGPU_FFT_FFT_HPP

/**
 * @file fft.hpp
 *
 * This header provides the mighty fft class
 */

#include <mgpu/core/rank_group.hpp>
#include <mgpu/backend/fft.hpp>
#include <mgpu/seg_dev_stream.hpp>

namespace mgpu
{

template <typename InputType, typename OutputType>
class fft
{
  private:
    typedef boost::array<backend::fft<InputType, OutputType> *,
                         MGPU_MAX_DEVICES>                       resource_type;
    typedef std::size_t                                          size_type;
    typedef boost::array<size_type, MGPU_MAX_DEVICES>            sizes_type;

  public:
    static const bool forward_possible  =
      backend::fft_traits<InputType, OutputType>::forward_possible::value;

    static const bool inverse_possible  =
      backend::fft_traits<InputType, OutputType>::inverse_possible::value;

  public:

    /// default constructor, do nothing
    fft() {}

    /// create fft object for 2D FFT
    explicit fft(std::size_t dim1, std::size_t dim2, std::size_t batch = 1,
      const rank_group & ranks = environment::get_all_ranks());

    /// set stream
    inline void set_stream(seg_dev_stream & stream);

    /// reset stream
    inline void reset_stream();

    /// destroy fft object
    ~fft();

    /// calculate forward FFT
    template <typename InputRange, typename OutputRange>
    void forward(InputRange & in, OutputRange & out)
    {
      forward_impl(in, out);
    }

    /// calculate inverse FFT
    template <typename InputRange, typename OutputRange>
    void inverse(InputRange & in, OutputRange & out)
    {
      inverse_impl(in, out);
    }

  private:

    /// implementation of forward FFT
    template <typename InputRange, typename OutputRange>
    void forward_impl(InputRange & in, OutputRange & out);

    /// implementation of inverse FFT
    template <typename InputRange, typename OutputRange>
    void inverse_impl(InputRange & in, OutputRange & out);

  private:

    /// blas handles, each for one device
    resource_type resources_;

    /// device group
    rank_group ranks_;

    /// batchsize
    size_type batch_;

    /// number of segments the class was created for
    size_type segments_;

    /// blocks in each segment
    sizes_type blocks_;
};

} // namespace mgpu

#include <mgpu/fft/detail/fft.ipp>

#endif // MGPU_FFT_FFT_HPP
