// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_SEG_DEV_STREAM_HPP
#define MGPU_SEG_DEV_STREAM_HPP

/**
 * @file seg_dev_stream.hpp
 *
 * This header provides the segmented streams
 */

#include <mgpu/backend/dev_stream.hpp>
#include <mgpu/backend/dev_management.hpp>
#include <mgpu/environment.hpp>
#include <mgpu/invoke.hpp>
#include <mgpu/core/ref.hpp>
#include <mgpu/core/rank_group.hpp>

namespace mgpu
{

namespace detail
{
  inline void create_dev_stream_impl(backend::dev_stream * & s)
  { s->create(); }

  inline void destroy_dev_stream_impl(backend::dev_stream * & s)
  { delete s; }

} // namespace details


/**
 * @brief a segmented device stream handle, holding one stream for each device
 */
struct seg_dev_stream
{
  /// create streams
  inline seg_dev_stream() : default_(false)
  {
    const rank_group & ranks = environment::get_all_ranks();
    for(std::size_t rank=0; rank<ranks.size(); rank++)
    {
      // we allocate the object
      streams_[rank] = new backend::dev_stream(true);
      // but do the device specific stuff in the thread
      invoke(detail::create_dev_stream_impl, streams_[rank], ranks[rank]);
    }
  }

  /// create default stream
  explicit inline seg_dev_stream(int) : default_(true)
  {
    for(std::size_t rank=0; rank<MGPU_NUM_DEVICES; rank++)
    {
      streams_[rank] =
        const_cast<backend::dev_stream *>(&backend::default_stream);
    }
  }

  /// destroy streams
  inline ~seg_dev_stream()
  {
    // we don't have to do anything if this is a default stream
    if(default_)
    {
      return;
    }
    else
    {
      const rank_group & ranks = environment::get_all_ranks();
      for(std::size_t rank=0; rank<ranks.size(); rank++)
      {
        invoke(detail::destroy_dev_stream_impl, streams_[rank], ranks[rank]);
      }
    }
  }

  /// operator [], accessing streams
  backend::dev_stream const & operator [](const int & offset) const
  { return * streams_[offset]; }

  /// operator [], accessing streams
  backend::dev_stream const & operator [](const int & offset)
  { return * streams_[offset]; }

  private:

    /// pointers to device streams
    boost::array<backend::dev_stream *, MGPU_NUM_DEVICES> streams_;

    /// indicates if stream is default stream
    bool default_;
};

extern const seg_dev_stream default_seg_stream;

} // namespace mgpu


#endif // MGPU_SEG_DEV_STREAM_HPP
