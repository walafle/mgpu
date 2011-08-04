// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_SEG_DEV_VECTOR_HPP
#define MGPU_CONTAINER_SEG_DEV_VECTOR_HPP

/**
 * @file seg_dev_vector.hpp
 *
 * This header contains the segmented device vector.
 */

#include <boost/array.hpp>

#include <mgpu/core/dev_id.hpp>
#include <mgpu/core/rank_group.hpp>
#include <mgpu/container/clone_size.hpp>
#include <mgpu/container/dev_vector.hpp>
#include <mgpu/container/seg_dev_iterator.hpp>
#include <mgpu/environment.hpp>
#include <mgpu/config.hpp>
#include <mgpu/container/detail/seg_dev_data_base.hpp>

namespace mgpu
{

template <typename Value, typename Alloc>
class seg_dev_iterator;

template<typename T, typename Alloc = dev_allocator<T> >
class seg_dev_vector :
  public detail::seg_dev_data_base<T, typename Alloc::pointer>
{
  private:
    typedef detail::seg_dev_data_base<T, typename Alloc::pointer> base;

  public:

    typedef typename base::size_type                      size_type;
    typedef typename base::sizes_type                     sizes_type;
    typedef Alloc                                         allocator_type;

    typedef typename Alloc::pointer                       pointer;
    typedef const typename Alloc::pointer                 const_pointer;

    typedef typename base::local_range                    local_range;
    typedef typename base::const_local_range              const_local_range;

    typedef typename base::local_iterator                 local_iterator;
    typedef typename base::const_local_iterator           const_local_iterator;

    typedef seg_dev_iterator<T, Alloc>                    seg_iterator;
    typedef const seg_dev_iterator<T, Alloc>              const_seg_iterator;

  private:
    typedef typename base::storage_type                   storage_type;

  public:

    /**
     * @brief construct an empty vector
     */
    seg_dev_vector();

    /**
     * @brief create segmented vector of size on the specified devices
     *
     * Constructor allocates a vector of size @p size and distributes it
     * equally across devices
     *
     * @param size size of entire vector that should be allocated
     *
     * @param ranks device group that specifies across which devices the
     * vector should be allocated
     */
    explicit seg_dev_vector(size_type size,
      const rank_group & ranks = environment::get_all_ranks());

    /**
     * @brief create segmented vector on specified devices, each segment is
     * has the specified size
     *
     * @param segmentsize size of one segment
     *
     * @param ranks device group that specifies across which devices the
     * vector should be allocated
     */
    explicit seg_dev_vector(clone_size segmentsize,
      const rank_group & ranks = environment::get_all_ranks());

    /**
     * @brief create segmented vector of size
     *
     * Constructor allocates a vector of size @p size and distributes it
     * equally across devices. The smallest unit to split the vector is the
     * blocksize.
     *
     * @param size size of entire vector that should be allocated
     *
     * @param blocksize smallest size of vector on one device, sub-vectors can
     * only be multiples of blocksize
     *
     * @param ranks device group that specifies across which devices the
     * vector should be allocated
     */
    explicit seg_dev_vector(size_type size, size_type blocksize,
      const rank_group & ranks = environment::get_all_ranks());

    /**
     * @brief create segmented vector of size on the specified devices
     *
     * Constructor allocates a vector of size @p size and distributes it
     * equally across devices
     *
     * @param size size of entire vector that should be allocated
     *
     * @param ranks device group that specifies across which devices the
     * vector should be allocated
     */
    static seg_dev_vector<T, Alloc> split(size_type ,
      const rank_group & ranks = environment::get_all_ranks());

    /**
     * @brief create segmented vector of size
     *
     * Constructor allocates a vector of size @p size and distributes it
     * equally across devices. The smallest unit to split the vector is the
     * blocksize.
     *
     * @param size size of entire vector that should be allocated
     *
     * @param blocksize smallest size of vector on one device, sub-vectors can
     * only be multiples of blocksize
     *
     * @param ranks device group that specifies across which devices the
     * vector should be allocated
     */
    static seg_dev_vector<T, Alloc> split(size_type size, size_type blocksize,
      const rank_group & ranks = environment::get_all_ranks());

    /**
     * @brief create segmented vector on specified devices, each segment is
     * has the specified size
     *
     * Named constructor allocates a vector of size ranks * @p size and
     * distributes it equally across devices
     *
     * @param segmentsize size of one segment
     *
     * @param ranks device group that specifies across which devices the
     * vector should be allocated
     */
    static seg_dev_vector<T, Alloc> clone(size_type segmentsize,
      const rank_group & ranks = environment::get_all_ranks());

    /// destroy the vector
    ~seg_dev_vector();

    /// return iterator to beginning of segmented device vector
    seg_iterator begin() { return seg_iterator(*this); }

    /// return constant iterator to beginning of segmented device vector
    const_seg_iterator begin() const { return seg_iterator(*this); }

    /// return iterator to end of segmented device vector
    seg_iterator end() { return seg_iterator(*this, size_); }

    /// return constant iterator to end of segmented device vector
    const_seg_iterator end() const { return seg_iterator(*this, size_); }

    /// fill with set_null bytes
    void set_null();

  private:

    /// helper to alloate the vector
    void allocate();

  private:

    /// reset the arrays in the vector
    void reset_arrays();

    /// reset all normal members
    void reset_scalar_members();

  private:

    using base::storage_;
    using base::size_;
    using base::overall_blocks_;
    using base::segments_;
    using base::blocksize_;
    using base::sizes_;
    using base::blocks_;
    using base::ranks_;
};


} // namespace mgpu

#include <mgpu/container/detail/seg_dev_vector.ipp>

#endif // MGPU_CONTAINER_SEG_DEV_VECTOR_HPP
