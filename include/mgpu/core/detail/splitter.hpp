// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_DETAIL_SPLITTER_HPP
#define MGPU_CORE_DETAIL_SPLITTER_HPP

/**
 * @file splitter.hpp
 *
 * This header contains the splitter helper class
 */

#include <mgpu/exception.hpp>

namespace mgpu
{

namespace detail
{


/**
 * @brief splitter algorithm to distribute a vector in blocks across devices
 *
 * @ingroup container
 */
class splitter
{
  private:
    std::size_t split_;
    std::size_t slice_large_;
    std::size_t slice_small_;
    int num_slices_per_split_rest_;

  public:
    /**
     * @brief constructor
     *
     * @param overall_size size of vector that should be split
     *
     * @param blocksize size of blocks the vector should be split in
     *
     * @param split number of units the vector should be split in
     */
    inline splitter(const std::size_t & overall_size,
      const std::size_t & blocksize, const unsigned int & split) :
        split_(split), slice_large_(0),
        slice_small_(0), num_slices_per_split_rest_(0)
    {
      if(overall_size % blocksize != 0)
      {
        MGPU_THROW("incompatible vector sizes");
      }

      if(split < 1)
      {
        MGPU_THROW("split parameter < 1");
      }

      if(overall_size < blocksize)
      {
        MGPU_THROW("overall size smallter than blocksize\n");
      }

      const unsigned int num_slices = overall_size / blocksize;
      const unsigned int num_slices_per_split = num_slices / split_;
      num_slices_per_split_rest_ = num_slices % split_;

      // if split > num_slices
      if(num_slices_per_split == 0)
      {
        split_ = num_slices_per_split_rest_;
      }
      // calculate large and small sizes (large is with rest, small without)
      slice_small_ = num_slices_per_split * blocksize;
      slice_large_ =
        slice_small_ + ((num_slices_per_split_rest_ > 0) ? blocksize : 0);
      // increment once since we decrement before we check
      num_slices_per_split_rest_++;
    }

    /**
     * @brief return size of split vector
     *
     * Operator returns the size of the split vector depending on the iteration,
     * e.g. the first call returns the size of the first split vector and the
     * n-th call returns the n-th size of the vector
     *
     * @return size of the current vector
     */
    inline std::size_t operator ++(int)
    {
      if(split_ == 0)
      {
        return 0;
      }

      // count how often the operator was called
      split_--;
      // count how many large sizes we returned
      num_slices_per_split_rest_--;

      if(num_slices_per_split_rest_ > 0)
      {
        return slice_large_;
      }
      else
      {
        return slice_small_;
      }
    }

};


} // namespace detail

} // namespace mgpu


#endif // MGPU_CORE_DETAIL_SPLITTER_HPP
