// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_SYNCHRONIZATION_HPP
#define MGPU_SYNCHRONIZATION_HPP

/**
 * @file synchronization.hpp
 *
 * This header contains all synchronization functions
 */


#include <mgpu/core/dev_rank.hpp>
#include <mgpu/core/rank_group.hpp>
#include <mgpu/core/detail/runtime.hpp>
#include <mgpu/seg_dev_stream.hpp>

namespace mgpu
{

extern const bool blocking;
extern const bool non_blocking;

/**
 * @brief instruct all device queues to wait till all other device queues are
 * empty, optionally blocks caller
 *
 * The queue may have launched asynchronous operations that return early. This
 * call does not ensure that those operations are finished.
 *
 * @param type type of synchronization with respect to the calling thread,
 * blocking: block calling thread, non_blocking: don't block calling thread
 */
inline void barrier(const bool type = blocking)
{
  if(type == blocking)
  {
    detail::runtime::insert_global_barrier_();
  }
  else
  {
    detail::runtime::insert_internal_barrier_();
  }
}

/**
 * @brief caller blocks until device queue is empty
 *
 * The queue may have launched asynchronous operations that return early. This
 * call does not ensure that those operations are finished. This call implicitly
 * blocks the calling thread since different behavior is nonsensical.
 *
 * @param rank rank of device to synchronize with
 */
inline void barrier(const dev_rank_t & rank)
{
  detail::runtime::insert_barrier_(rank);
}


/**
 * @brief instruct all device queues to wait for asynchronous operations to
 * finish
 */
inline void synchronize()
{
  detail::runtime::insert_global_sync_();
}

/**
 * @brief instruct all device queues to wait for asynchronous operations in
 * stream to finish
 *
 * @param stream to synchronize on
 */
inline void synchronize(const seg_dev_stream & streams)
{
  const rank_group & ranks = environment::get_all_ranks();
  for(unsigned int i=0; i<ranks.size(); i++)
  {
    detail::runtime::insert_sync_stream_(streams[i], ranks[i]);
  }
}

/**
 * @brief instruct device queue to wait for asynchronous operations to finish
 *
 * @param rank rank of device to insert barrier in
 */
inline void synchronize(const dev_rank_t & rank)
{
  detail::runtime::insert_sync_(rank);
}

/**
 * @brief instruct device queues in group to wait for asynchronous operations to
 * finish
 *
 * @param group group of devices to insert barrier in
 */
inline void synchronize(const rank_group & ranks)
{
  for(unsigned int i=0; i<ranks.size(); i++)
  {
    detail::runtime::insert_sync_(ranks[i]);
  }
}

/**
 * @brief instruct device queues in group to wait for asynchronous operations to
 * finish
 *
 * @param group group of devices to insert barrier in
 * @param stream to synchronize on
 */
inline void synchronize(const seg_dev_stream & streams,
  const rank_group & ranks)
{
  for(unsigned int i=0; i<ranks.size(); i++)
  {
    dev_rank_t r = ranks[i];
    detail::runtime::insert_sync_stream_(streams[r], r);
  }
}

/**
 * @brief instruct all device queues to wait for all asynchronous operations to
 * finish, then have them wait until all other device queues are empty,
 * optionally blocks caller
 *
 * The queue may have launched asynchronous operations that return early. This
 * call ensure that those operations are finished.
 *
 * @param type type of synchronization with respect to the calling thread,
 * blocking: block calling thread, non_blocking: don't block calling thread
 */
inline void synchronize_barrier(const bool type = blocking)
{
  synchronize();
  barrier(blocking);
}


/**
 * @brief instruct all device queues to wait for all asynchronous operations in
 * a stream to finish, then have them wait until all other device queues are
 * empty, optionally blocks caller
 *
 * The queue may have launched asynchronous operations that return early. This
 * call ensure that those operations are finished.
 *
 * @param stream to synchronize on
 * @param type type of synchronization with respect to the calling thread,
 * blocking: block calling thread, non_blocking: don't block calling thread
 */
inline void synchronize_barrier(const seg_dev_stream & streams,
  const bool type = blocking)
{
  synchronize(streams);
  barrier(blocking);
}

/**
 * @brief instruct device queue to wait for all asynchronous operations to
 * finish, caller blocks until this is done
 *
 * The queue may have launched asynchronous operations that return early. This
 * call ensure that those operations are finished.
 *
 * @param rank rank of device to synchronize with
 */
inline void synchronize_barrier(const dev_rank_t & rank)
{
  synchronize(rank);
  barrier(rank);
}




} // namespace mgpu



#endif // MGPU_SYNCHRONIZATION_HPP
