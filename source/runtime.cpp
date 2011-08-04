// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#include <mgpu/core/detail/runtime.hpp>
#include <mgpu/exception.hpp>
#include <mgpu/backend/backend.hpp>

#ifdef _MSC_VER
#include <Windows.h>
#endif

namespace mgpu
{
namespace detail
{

 void dummy_function() {}

// init, finalize
// _____________________________________________________________________________

void runtime::init(const dev_group & devices)
{
  if(initialized_)
  {
    MGPU_THROW("environment already initialized!");
  }
  devices_ = devices;

  global_barrier = new boost::barrier(devices_.size()+1);
  if(devices_.size() > 1)
  {
    internal_barrier = new boost::barrier(devices_.size());
  }

  device_to_rank_.set_size_(devices_.size());

  for(unsigned int i=0; i<devices_.size(); i++)
  {
    device_to_rank_[devices_[i]] = i;
    single_barriers_[i] = new boost::barrier(2);
    shutdown_flags_[i] = false;
    queues_[i] = new MGPU_RUNTIME_QUEUE_TYPE<boost::function<void()> >();
    threads_[i] = new boost::thread(dev_threadloop, i, devices_[i]);
    // add barrier here to let device runtime setup device
    insert_barrier_(i);
  }

  all_ranks_ = dev_group::from_to(0, devices_.size());
  initialized_ = true;
}


void runtime::finalize()
{
  if(!initialized_)
  {
    MGPU_THROW("environment not initialized!");
  }

  for(unsigned int i=0; i<devices_.size(); i++)
  {
    shutdown_flags_[i] = true;
  }

#if defined(__GNUC__)
    asm volatile("" ::: "memory");
#elif defined(_MSC_VER)
    MemoryBarrier();
#else
    #error "this backend is currently not supported!"
#endif

  invoke_all_devices(dummy_function);

  for(unsigned int i=0; i<devices_.size(); i++)
  {
    threads_[i]->join();
  }

  for(unsigned int i=0; i<devices_.size(); i++)
  {
    delete threads_[i];
    delete queues_[i];
    delete single_barriers_[i];
  }

  delete global_barrier;
  if(devices_.size() > 1)
  {
    delete internal_barrier;
  }

  initialized_ = false;
}

// threadloop
// _____________________________________________________________________________

void runtime::dev_threadloop(int tid, dev_id_t id)
{
  backend::set_dev(id);
  backend::sync_dev();
  boost::function<void()> task;
  while(true)
  {
    volatile bool shutdown = shutdown_flags_[tid];
#if defined(__GNUC__)
    asm volatile("" ::: "memory");
#elif defined(_MSC_VER)
    MemoryBarrier();
#else
    #error "this backend is currently not supported!"
#endif
    volatile bool havetask = queues_[tid]->consume(task, !shutdown);

    if(havetask)
    {
      task();
    }
    else if(shutdown)
    {
      backend::reset_dev();
      return;
    }
    else
    {
      boost::this_thread::yield();
    }
  }
}

// synchronization methods
// _____________________________________________________________________________

void runtime::insert_internal_barrier_()
{
  if(devices_.size() > 1)
  {
    invoke_all_devices(boost::bind(barrier_, internal_barrier));
  }
}

void runtime::insert_global_barrier_()
{
  invoke_all_devices(boost::bind(barrier_, global_barrier));
  global_barrier->wait();
}

void runtime::insert_barrier_(const dev_rank_t & rank)
{
  invoke_device(boost::bind(barrier_, single_barriers_[rank]), rank);
  single_barriers_[rank]->wait();
}

void runtime::insert_global_sync_()
{
  invoke_all_devices(sync_);
}

void runtime::insert_sync_(const dev_rank_t & rank)
{
  invoke_device(sync_, rank);
}

void runtime::insert_sync_stream_(const ::mgpu::backend::dev_stream & stream,
  const dev_rank_t & rank)
{
  invoke_device(boost::bind(sync_stream_, &stream), rank);
}

std::size_t runtime::size()
{
  return all_ranks_.size();
}

// invoke, sync and fence
// _____________________________________________________________________________


void runtime::invoke_all_devices(const boost::function<void()> & function)
{
  for(unsigned int i=0; i<devices_.size(); i++)
  {
    queues_[i]->produce(function);
  }
}


void runtime::invoke_device(const boost::function<void()> & function,
  const dev_rank_t & rank)
{
  if(rank >= (int)devices_.size())
  {
    MGPU_THROW("Invalid device offset in invoke_device");
  }
  queues_[rank]->produce(function);
}


void runtime::barrier_(boost::barrier * const b)
{
  b->wait();
}

void runtime::sync_()
{
  backend::sync_dev();
}

void runtime::sync_stream_(::mgpu::backend::dev_stream const * stream)
{
  backend::sync_dev(*stream);
}

void runtime::sync_barrier_(boost::barrier * const b)
{
  backend::sync_dev();
  b->wait();
}

void runtime::sync_stream_barrier_(::mgpu::backend::dev_stream const * stream,
  boost::barrier * const b)
{
  backend::sync_dev(*stream);
  b->wait();
}

// static members
// _____________________________________________________________________________

boost::array<MGPU_RUNTIME_QUEUE_TYPE<boost::function<void()> > *,
  MGPU_MAX_DEVICES> runtime::queues_;

boost::array<boost::thread *, MGPU_MAX_DEVICES> runtime::threads_;

boost::array<bool, MGPU_MAX_DEVICES> runtime::shutdown_flags_;

boost::array<boost::barrier *, MGPU_MAX_DEVICES> runtime::single_barriers_;

boost::barrier * runtime::global_barrier;

boost::barrier * runtime::internal_barrier;

dev_group runtime::devices_;

rank_group runtime::device_to_rank_;

rank_group runtime::all_ranks_;

bool runtime::initialized_;

} // namespace detail

} // namespace mgpu

