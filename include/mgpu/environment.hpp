// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_ENVIRONMENT_HPP
#define MGPU_ENVIRONMENT_HPP

/**
 * @file environment.hpp
 *
 * This header contains the environment class which provides routines to
 * initialize, query and finalize the runtime
 */

#include <stdio.h>
#include <mgpu/config.hpp>
#include <mgpu/core/dev_group.hpp>
#include <mgpu/core/detail/runtime.hpp>

namespace mgpu
{

/**
 * @brief runtime, an all private class, is a collection of all runtime related
 * variables and data
 */
class environment
{
  public:

    /**
     * @brief construct the environment
     *
     * @param devices devices that shall be used
     */
    explicit inline environment(const dev_group & devices = all_devices)
    {
      detail::runtime::init(devices);
    }

    /**
     * @brief destructor
     */
    inline ~environment()
    {
      detail::runtime::finalize();
    }

    /**
     * @brief return size of environment
     *
     * @return size
     */
    inline std::size_t size()
    {
      return detail::runtime::devices_.size();
    }

    /**
     * @brief return size of environment
     *
     * @return size
     */
    static inline std::size_t get_size()
    {
      return detail::runtime::devices_.size();
    }

    /**
     * @brief return devices
     *
     * @return devices
     */
    inline const dev_group & devices()
    {
      return detail::runtime::devices_;
    }

    /**
     * @brief return devices
     *
     * @return devices
     */
    static inline const dev_group & get_devices()
    {
      return detail::runtime::devices_;
    }

    /**
     * @brief return a device group that contains all devices that are used
     *
     * @return group of ranks
     */
    inline const rank_group & all_ranks()
    {
      return detail::runtime::all_ranks_;
    }

    /**
     * @brief return a device group that contains all devices that are used
     *
     * @return group of ranks
     */
    static inline const rank_group & get_all_ranks()
    {
      return detail::runtime::all_ranks_;
    }

    /**
     * @brief return the device id for a given device rank
     *
     * @return device id
     */
    static inline const dev_id_t & dev_id(const dev_rank_t & rank)
    {
      return detail::runtime::devices_[rank];
    }

    /**
     * @brief return the device rank to a given device id
     *
     * @return device rank
     */
    static inline const dev_rank_t & rank(const dev_id_t & id)
    {
      printf("env rank id %d\n", id);
      return detail::runtime::device_to_rank_[id];
    }

};


} // namespace mgpu


#endif // MGPU_CORE_RUNTIME_HPP
