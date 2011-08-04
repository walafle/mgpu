// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_TRANSFER_P2P_ENABLER_HPP
#define MGPU_TRANSFER_P2P_ENABLER_HPP

/**
 * @file p2p.hpp
 *
 * This header provides the p2p communication functionality
 */

#include <mgpu/core/rank_group.hpp>
#include <mgpu/core/dev_id.hpp>
#include <mgpu/environment.hpp>
#include <mgpu/backend/dev_management.hpp>
#include <mgpu/synchronization.hpp>
#include <mgpu/invoke.hpp>

namespace mgpu
{

namespace details
{
  inline void enable_p2p(rank_group & ranks, dev_rank_t rank)
  {
    for(std::size_t i=0; i<ranks.size(); i++)
    {
      if(i == (unsigned)rank)
      {
        continue;
      }
      backend::enable_p2p(environment::dev_id(ranks[i]));
    }
  }

  inline void disable_p2p(rank_group & ranks, dev_rank_t rank)
  {
    for(std::size_t i=0; i<ranks.size(); i++)
    {
      if(i == (unsigned)rank)
      {
        continue;
      }
      backend::disable_p2p(environment::dev_id(ranks[i]));
    }
  }
}

/**
 * @brief enable peer to peer communication on all devices between all devices
 *
 * Constructor enables P2P, destructor disables P2p
 *
 * @ingroup communication
 */
struct p2p_enabler
{
  inline p2p_enabler(const rank_group & ranks = environment::get_all_ranks()) :
    ranks_(ranks)
  {
    for(std::size_t rank=0; rank<ranks_.size(); rank++)
    {
      invoke(details::enable_p2p, ranks_, rank, ranks_[rank]);
      synchronize_barrier();
    }
  }

  inline ~p2p_enabler()
  {
    for(std::size_t rank=0; rank<ranks_.size(); rank++)
    {
      invoke(details::disable_p2p, ranks_, rank, rank);
      synchronize_barrier();
    }
  }

  private:
    rank_group ranks_;
};

} // namespace mgpu


#endif // MGPU_TRANSFER_P2P_ENABLER_HPP
