// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_RUNTIME_HPP
#define MGPU_CORE_RUNTIME_HPP

/**
 * @file runtime.hpp
 *
 * Contains the runtime components of the library.
 */

#include <boost/function.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/array.hpp>

#include <mgpu/config.hpp>
#include <mgpu/core/dev_group.hpp>
#include <mgpu/core/rank_group.hpp>
#include <mgpu/core/detail/queue_lockfree.hpp>
#include <mgpu/core/detail/queue_mutex.hpp>
#include <mgpu/backend/dev_stream.hpp>

namespace mgpu
{
  class environment;
}

namespace mgpu
{
namespace detail
{

/**
 * @brief runtime, an all private class, is a collection of all runtime related
 * variables and data
 */
class runtime
{
    friend class ::mgpu::environment;

  public:

    /**
     * @brief enqueue a function for all devices
     *
     * @param function that should be enqueued
     */
    static void invoke_all_devices(const boost::function<void()> & function);

    /**
     * @brief enqueue a function for one device
     *
     * @param function that should be enqueued
     */
    static void invoke_device(const boost::function<void()> & function,
      const dev_rank_t & rank);

    // _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _

    /**
     * @brief insert a barrier in all device queues
     */
    static void insert_internal_barrier_();

    /**
     * @brief insert a barrier in all device queues; include caller
     */
    static void insert_global_barrier_();

    /**
     * @brief insert a barrier in device specified by rank; include caller
     */
    static void insert_barrier_(const dev_rank_t & rank);

    /**
     * @brief insert device synchronization in device specified by rank
     */
    static void insert_global_sync_();

    /**
     * @brief insert device synchronization in device specified by rank
     */
    static void insert_sync_(const dev_rank_t & rank);

    /**
     * @brief insert stream synchronization in device specified by rank
     */
    static void insert_sync_stream_(const ::mgpu::backend::dev_stream & stream,
      const dev_rank_t & rank);

    /**
     * @brief return the number of ranks
     */
    static std::size_t size();

    // _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _

  private:

    /**
     * @brief create the runtime with the devices specified here
     * @param devices
     */
    static void init(const dev_group & devices = all_devices);

    /**
     * @brief tear down runtime
     */
    static void finalize();

    /**
     * @brief function that is the device thread
     */
    static void dev_threadloop(int tid, dev_id_t id);

    /**
     * @brief check if runtime is initialized
     * @return true if runtime is initialized
     */
    static inline bool is_initialized() { return runtime::initialized_; }

    // _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _

    /**
     * @brief synchronization method that can be added to thread queues
     *
     * @param b pointer to barrier to synchronize with
     */
    static void barrier_(boost::barrier * const b);

    /**
     * @brief fence method that can be added to thread queues
     */
    static void sync_();

    /**
     * @brief fence stream method that can be added to thread queues
     *
     * @param stream pointer to stream to synchronize
     */
    static void sync_stream_(::mgpu::backend::dev_stream const * stream);

    /**
     * @brief combined synchronization and fence method
     *
     * @param b pointer to barrier to synchronize with
     */
    static void sync_barrier_(boost::barrier * const b);

    /**
     * @brief combined stream synchronization and fence method
     *
     * @param stream pointer to stream to synchronize
     * @param b pointer to barrier to synchronize with
     */
    static void sync_stream_barrier_(::mgpu::backend::dev_stream const * stream,
      boost::barrier * const b);

    // _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _

    /// hold the task queues
    static boost::array<MGPU_RUNTIME_QUEUE_TYPE<boost::function<void()> > *,
      MGPU_MAX_DEVICES> queues_;

    /// hold the threads
    static boost::array<boost::thread *, MGPU_MAX_DEVICES> threads_;

    /// hold shutdown flags
    static boost::array<bool, MGPU_MAX_DEVICES> shutdown_flags_;

    /// barrier for global synchronization (size = devices_.size() + 1)
    static boost::barrier * global_barrier;

    /// barrier for internal synchronization (size = devices_.size())
    static boost::barrier * internal_barrier;

    /// barriers, one for each device for synchronization with calling thread
    static boost::array<boost::barrier *, MGPU_MAX_DEVICES> single_barriers_;

    /// devices that are used in environment
    static dev_group devices_;

    /// ranks that are available in environment
    static rank_group device_to_rank_;

    /// ranks that are available in environment
    static rank_group all_ranks_;

    /// has environment been initialized
    static bool initialized_;
};

} // namespace detail

} // namespace mgpu


#endif // MGPU_CORE_RUNTIME_HPP
