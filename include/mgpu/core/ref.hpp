// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CORE_REF_HPP
#define MGPU_CORE_REF_HPP

/**
 * @file ref.hpp
 *
 * This header provides a generic device reference class
 *
 * based on Boost.Ref library:
 * Copyright (C) 1999, 2000 Jaakko Jarvi (jaakko.jarvi@cs.utu.fi)
 * Copyright (C) 2001, 2002 Peter Dimov
 * Copyright (C) 2002 David Abrahams
 * See http://www.boost.org/libs/bind/ref.html for documentation.
 */

#include <stddef.h>

#include <boost/utility/addressof.hpp>

#include <mgpu/core/dev_ptr.hpp>

namespace mgpu
{

/**
 * @brief a generic device reference class that can store any reference to
 * device resources
 *
 * @ingroup core
 */
template<class T>
struct ref
{
  public:

    typedef T type;

    ref(): t_(NULL) {}
    explicit ref(T & t): t_(&t) {}
    explicit ref(T * t): t_(t) {}
    // explicit ref(const T& t): t_(const_cast<T*>(&t)) {}

    operator T& () const { return *t_; }
    T& get() const { return *t_; }
    T* get_pointer() const { return t_; }

    void set(T * t) { t_ = t; }
    void set(T & t) { *t_ = t; }

    void reset() { t_ = NULL; }

  private:

    T* t_;
};


} // namespace mgpu


#endif // MGPU_CORE_REF_HPP
