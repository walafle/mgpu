// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_DEV_SET_SCOPED_HPP
#define MGPU_CORE_DEV_SET_SCOPED_HPP

/** 
 * @file dev_set_scoped.hpp
 *
 * This header contains the dev_set_scoped class
 */
#include <mgpu/core/dev_rank.hpp>
#include <mgpu/core/dev_id.hpp>
#include <mgpu/backend/dev_management.hpp>

namespace mgpu
{

/**
 * @brief allows setting the device for a scope, automatically resets the scope
 * 
 * @ingroup core
 */
class dev_set_scoped
{
#ifndef MGPU_DISALLOW_SET_SCOPED

  public:
    explicit inline dev_set_scoped(const dev_id_t & id)
      : buffered_id(backend::get_dev()), current_id(buffered_id)
    { set(id); }

    inline ~dev_set_scoped() { set(buffered_id); }

    inline void set(const dev_id_t & id)
    {
      if(id != current_id)
      {
        backend::set_dev(id);
        current_id = id;
      }
    }

  private:
    const dev_id_t buffered_id;
    dev_id_t current_id;

#else

  // make class do nothing in this case
  public:
    explicit inline dev_set_scoped(const dev_id_t &) {}
    inline ~dev_set_scoped() {}
    inline void set(const dev_id_t &) {}

#endif

};


} // namespace mgpu


#endif // MGPU_CORE_DEV_SET_SCOPED_HPP
