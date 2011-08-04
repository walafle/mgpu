// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_CONTAINER_CLONE_SIZE_HPP
#define MGPU_CONTAINER_CLONE_SIZE_HPP

/**
 * @file clone_size.hpp
 */


namespace mgpu
{

/// class that holds a size and indicates that it should be cloned
struct clone_size
{
  public:

    typedef std::size_t size_type;

  public:

    /// create a clone_size object with a size
    inline clone_size(const size_type & size) : size_(size) { }

    /// access the size that is stored in the object
    inline const size_type & get_size() const { return size_; }

  private:

    /// size stored in object
    size_type size_;

};


} // namespace mgpu

#endif // MGPU_CONTAINER_CLONE_SIZE_HPP
