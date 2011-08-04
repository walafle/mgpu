// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_DETAIL_QUEUE_LOCKFREE_HPP
#define MGPU_CORE_DETAIL_QUEUE_LOCKFREE_HPP

#ifdef _MSC_VER
#include <Windows.h>
#endif

/**
 * @file queue_lockfree.hpp
 *
 * This header contains the queue_lockfree class which can be used as the
 * storage type for the environment class.
 */

namespace mgpu
{

/**
 * @brief Represents a lockfree queue
 *
 * This class represents a lockfree queue. It can be used as an option for the
 * environment class. It is based on an article by Herb Sutter:
 * http://www.drdobbs.com/high-performance-computing/210604448
 *
 * Atomics used in the code are replaced by GCC built-in instructions and manual
 * memory barriers.
 *
 * The queue can handle one producing thread and one consuming thread. Copying a
 * queue is not possible. The queue is designed in a first in first out manner.
 *
 * @tparam T type that should be stored in the queue
 */
template <typename T>
class queue_lockfree
{
  private:
    /// one single queue element contains value and a pointer to next node
    struct Node
    {
      Node(T val) : value(val), next(NULL) { }
      T value;
      Node* next;
    };

    /// point to first node (for producer only)
    Node* first;
    /// point to first unconsumed node (shared)
    Node* divider;
    /// point to last unconsumed node (shared)
    Node* last;

  public:

    /**
     * @brief Create a lockfree queue
     */
    queue_lockfree()
    {
      first = new Node(T());
      divider = first;
      last = first;
    }

    /**
     * @brief Destroy the lockfree queue
     */
    ~queue_lockfree()
    {
      // release the list
      while(first != NULL)
      {
        Node* tmp = first;
        first = tmp->next;
        delete tmp;
      }
    }

    /**
     * @brief Produce an entry in the queue
     *
     * Produce queue element, destroy queue elements that are already used
     *
     * @param t element that should be added to the queue
     */
    void produce(const T& t)
    {
      if(first != divider)
      {
        // reuse old item
        last->next = first;
        first = first->next;
        *(last->next) = t;
      }
      else
      {
        // add the new item
        last->next = new Node(t);
      }

#if defined(__GNUC__)
      // prevend compiler reordering
      asm volatile("" ::: "memory");
      // gcc atomic, cast to void to prevent "value computed is not used"
      (void)__sync_lock_test_and_set(&last, last->next);
#elif defined(_MSC_VER)
      // msvc
      (void)InterlockedExchangePointer(
		  reinterpret_cast<void * volatile *>(&last), last->next);
#else
      #error "this backend is currently not supported!"
#endif

//      // trim unused nodes
//      while(first != divider)
//      {
//        Node* tmp = first;
//        first = first->next;
//        delete tmp;
//      }
    }

    /**
     * @brief Consume entry in lockfree queue
     *
     * Tests if there is an entry in the queue to consume then return it.
     *
     * @param result contains the contents of the oldest node if there is one.
     *
     * @return True if the queue is not empty, else false
     */
    bool consume(T& result, const bool)
    {
      // if queue is nonempty
      if(divider != last)
      {
        // copy it back
        result = divider->next->value;

#if defined(__GNUC__)
        // prevend compiler reordering
        asm volatile("" ::: "memory");
        // gcc atomic, cast to void to prevent "value computed is not used"
        (void)__sync_lock_test_and_set(&divider, divider->next);
#elif defined(_MSC_VER)
      // msvc
      (void)InterlockedExchangePointer(
		  reinterpret_cast<void * volatile *>(&divider), divider->next);
#else
        #error "this backend is currently not supported!"
#endif

        // report success
        return true;
      }
      // else report empty
      return false;
    }
};

} // namespace mgpu

#endif // MGPU_CORE_DETAIL_QUEUE_LOCKFREE_HPP
