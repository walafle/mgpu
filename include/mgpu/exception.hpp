// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_EXCEPTION_HPP
#define MGPU_EXCEPTION_HPP

/**
 * @file exception.hpp
 *
 * This header provides an exception classes that report CUDA errors to the user
 * as well as macros for shorthand use with CUDA runtime calls.
 */

#include <string>

#include <boost/exception/info.hpp>
#include <boost/exception/exception.hpp>
#include <boost/throw_exception.hpp>
#include <boost/exception/get_error_info.hpp>


namespace mgpu
{

/// holds an exception message
typedef boost::error_info<struct tag_cuda_err_text, std::string>
  exception_message;

/**
 * @brief mgpu exception, thrown in case of a general mgpu related error
 *
 * @ingroup main
 */
struct mgpu_exception : virtual boost::exception, virtual std::exception
{

  /**
   * @brief Return a well formated string containing exception description
   */
  virtual const char* what() const throw()
  {
    std::ostringstream msg;
    if(std::string const * mi = boost::get_error_info<exception_message>(*this))
    {
      msg << "MGPU exception:\n" << *mi << "\n";
    }
    if(const char* const* mi =
       boost::get_error_info< boost::throw_file >(*this))
    {
      msg << "In file: " << *mi;
    }
    if(int const * mi =
       boost::get_error_info< boost::throw_line >(*this))
    {
      msg << " (line " << *mi << ")\n";
    }
    if(const char* const* mi =
       boost::get_error_info< boost::throw_function >(*this))
    {
      msg << "In function: " << *mi << "\n";
    }
    return msg.str().c_str();
  }

};

} // namespace mgpu

#define MGPU_THROW(message)                                                    \
  throw boost::enable_current_exception(                                       \
      mgpu::mgpu_exception()) <<                                               \
    mgpu::exception_message(message) <<                                        \
    boost::throw_function(BOOST_CURRENT_FUNCTION) <<                           \
    boost::throw_file(__FILE__) <<                                             \
    boost::throw_line((int)__LINE__);

#endif // MGPU_EXCEPTION_HPP
