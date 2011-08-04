// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_TRANSFER_ENABLE_P2P_SCOPED_HPP
#define MGPU_TRANSFER_ENABLE_P2P_SCOPED_HPP

/** 
 * @file enable_p2p_scoped.hpp
 *
 * This header contains the enable_p2p_scoped class
 */

#include <mgpu/core/dev_id.hpp>
#include <mgpu/backend/dev_management.hpp>

namespace mgpu
{

/**
 * @brief allows enabling p2p communication in a scope
 * 
 * @ingroup core
 */
class enable_p2p_scoped
{
  // make class do nothing in this case
  public:
    explicit inline enable_p2p_scoped(const dev_id_t & to) : to_(to)
    { backend::enable_p2p(to_); }

    inline ~enable_p2p_scoped()
    { backend::disable_p2p(to_); }

  private:
    dev_id_t to_;
};


} // namespace mgpu


#endif // MGPU_TRANSFER_ENABLE_P2P_SCOPED_HPP
