// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_DETAIL_QUEUE_MUTEX_HPP
#define MGPU_CORE_DETAIL_QUEUE_MUTEX_HPP

/**
 * @file queue_mutex.hpp
 *
 * This header contains the queue_mutex class which can be used as the
 * storage type for the environment class.
 */

#include <queue>

#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>

namespace mgpu
{

/**
 * @brief Thread safe wrapper around std::queue
 *
 * @tparam T type that should be stored in the queue
 */
template <typename T>
class queue_mutex
{
  private:

    /// mutex to restrict access to queue to one entity
    boost::mutex mutex_;

    /// condition variable to handle empty queues
    boost::condition_variable cond_;

    /// queue that stores data
    std::queue<T> queue_;

  public:

    /**
     * @brief Create a mutex protected queue
     */
    queue_mutex() : mutex_(), queue_()
    { }

    /**
     * @brief Destroy the mutex protected queue
     */
    ~queue_mutex()
    { }

    /**
     * @brief Produce an entry in the queue
     *
     * Produce queue element, destroy queue elements that are already used
     *
     * @param t element that should be added to the queue
     */
    void produce(const T& t)
    {
      boost::lock_guard<boost::mutex> lock(mutex_);
      queue_.push(t);
      cond_.notify_one();
    }

    /**
     * @brief Consume entry in queue
     *
     * Tests if there is an entry in the queue to consume then return it. If
     * not return (if blocking = false) or block until something is in the queue
     * (blocking = true).
     *
     * @param result contains the contents of the oldest node if there is one.
     *
     * @return True if the queue is not empty, else false
     */
    bool consume(T& result, const bool blocking)
    {
      boost::unique_lock<boost::mutex> lock(mutex_);
      if(!blocking && queue_.empty())
      {
        return false;
      }
      while(queue_.empty())
      {
        cond_.wait(lock);
      }
      result = queue_.front();
      queue_.pop();
      return true;
    }
};

} // namespace mgpu

#endif // MGPU_CORE_DETAIL_QUEUE_MUTEX_HPP
