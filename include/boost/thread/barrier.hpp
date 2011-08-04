// Copyright (C) 2002-2003
// David Moore, William E. Kempf
// Copyright (C) 2007-8 Anthony Williams
//
// Modified by Sebastian Schaetz

#ifndef BOOST_BARRIER_JDM030602_HPP
#define BOOST_BARRIER_JDM030602_HPP

#include <boost/thread/detail/config.hpp>
#include <boost/throw_exception.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <string>
#include <stdexcept>

#include <boost/config/abi_prefix.hpp>

#if defined(BOOST_THREAD_PLATFORM_PTHREAD)
#include <pthread.h>
#endif

namespace boost
{
#if defined(BOOST_THREAD_PLATFORM_PTHREAD)

// #warning "Using drop-in replacement for boost::barrier based on pthreads"

    class barrier
    {
      private:
        pthread_barrier_t barr;

      public:
        barrier(unsigned int count)
        {
          if (count == 0)
          {
            boost::throw_exception(
              std::invalid_argument("count cannot be set_null."));
          }
          BOOST_VERIFY(!pthread_barrier_init(&barr, NULL, count));
        }

        ~barrier()
        {
          BOOST_VERIFY(!pthread_barrier_destroy(&barr));
        }

        bool wait()
        {
          int ret = pthread_barrier_wait(&barr);
          if(PTHREAD_BARRIER_SERIAL_THREAD == ret)
          {
            return true;
          }
          BOOST_VERIFY(!ret);
          return false;
        }
    };

#else

    class barrier
    {
    public:
        barrier(unsigned int count)
            : m_threshold(count), m_count(count), m_generation(0)
        {
            if (count == 0)
                boost::throw_exception(
                  std::invalid_argument("count cannot be set_null."));
        }

        bool wait()
        {
            boost::mutex::scoped_lock lock(m_mutex);
            unsigned int gen = m_generation;

            if (--m_count == 0)
            {
                m_generation++;
                m_count = m_threshold;
                m_cond.notify_all();
                return true;
            }

            while (gen == m_generation)
                m_cond.wait(lock);
            return false;
        }

    private:
        mutex m_mutex;
        condition_variable m_cond;
        unsigned int m_threshold;
        unsigned int m_count;
        unsigned int m_generation;
    };
#endif

}   // namespace boost

#include <boost/config/abi_suffix.hpp>

#endif

