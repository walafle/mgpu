// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BLAS_BLAS_HPP
#define MGPU_BLAS_BLAS_HPP

/**
 * @file blas.hpp
 *
 * This header provides the mighty blas class
 */

#include <mgpu/backend/blas.hpp>
#include <mgpu/core/rank_group.hpp>

namespace mgpu
{

class blas
{
  private:
    typedef boost::array<backend::blas *, MGPU_MAX_DEVICES>   resource_type;
    typedef std::size_t                                       size_type;

  public:

    /// construct segmented blas object
    explicit inline
    blas(const rank_group & ranks = environment::get_all_ranks());

    /// destroy segmented blas object
    inline ~blas();

    /// indicate that scalar values are passed as reference on host
    inline void set_scalar_device();

    /// indicate that scalar values are passed as reference on device
    inline void set_scalar_host();


    /// calculate inner product product
    template <typename XRange, typename YRange, typename ResultIterator>
    void inner_prod(XRange & x, YRange & y, ResultIterator result)
    {
      inner_prod_impl(x, y, result,
        typename ::mgpu::iterator_traits<ResultIterator>::segmented_tag());
    }

    /// calculate inner product (first vector is conjugated)
    template <typename XRange, typename YRange, typename ResultIterator>
    void inner_prod_c(XRange & x, YRange & y, ResultIterator result)
    {
      inner_prod_c_impl(x, y, result);
    }

    /**
     * @brief calculate y = a*x + y
     *
     * if alpha resides on the device, one alpha for each devices is required,
     * i.e. alpha must be a segmented device vector
     * if alpha resides on the host, a pointer to one alpha is expected
     */
    template <typename AlphaIterator, typename XRange, typename YRange>
    void axpy(AlphaIterator alpha, const XRange & x, YRange & y)
    {
      axpy_impl(alpha, x, y,
        typename ::mgpu::iterator_traits<AlphaIterator>::location_tag());
    }

  private:

    /// implementation of y = a*x + y where resides in device memory
    template <typename AlphaIterator, typename SegmentedXRange,
      typename SegmentedYRange>
    void axpy_impl(AlphaIterator alpha, const SegmentedXRange & x,
      SegmentedYRange & y, device_memory_tag);

    /// implementation of y = a*x + y where alpha resides on the host
    template <typename AlphaIterator, typename SegmentedXRange,
      typename SegmentedYRange>
    void axpy_impl(AlphaIterator alpha, const SegmentedXRange & x,
      SegmentedYRange & y, host_memory_tag);

    /// implementation of inner_prod
    template <typename XRange, typename YRange, typename ResultIterator>
    void inner_prod_impl(XRange & x, YRange & y, ResultIterator result,
      is_segmented_tag);

    template <typename XRange, typename YRange, typename ResultIterator>
    void inner_prod_impl(XRange & x, YRange & y, ResultIterator result,
      is_not_segmented_tag);

    /// implementation of inner_prod_c
    template <typename XRange, typename YRange, typename ResultIterator>
    void inner_prod_c_impl(XRange & x, YRange & y, ResultIterator result,
      is_segmented_tag);

    template <typename XRange, typename YRange, typename ResultIterator>
    void inner_prod_c_impl(XRange & x, YRange & y, ResultIterator result,
      is_not_segmented_tag);



  private:

    /// blas handles, each for one device
    resource_type resources_;

    /// device group
    rank_group ranks_;

    /// number of segments the class was created for
    size_type segments_;
};

} // namespace mgpu

#include <mgpu/blas/detail/blas.ipp>

#endif // MGPU_BLAS_BLAS_HPP
