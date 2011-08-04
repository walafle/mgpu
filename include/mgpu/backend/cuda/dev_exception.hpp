// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_BACKEND_CUDA_DEV_EXCEPTION_HPP
#define MGPU_BACKEND_CUDA_DEV_EXCEPTION_HPP

/**
 * @file dev_exception.hpp
 *
 * This header provides an exception classes that report CUDA errors to the user
 * as well as macros for shorthand use with CUDA runtime calls.
 */

#include <string>

#include <boost/exception/info.hpp>
#include <boost/exception/exception.hpp>
#include <boost/throw_exception.hpp>
#include <boost/exception/get_error_info.hpp>

#include <cuda_runtime.h>

namespace mgpu
{
namespace backend_detail
{
namespace cuda
{

/// holds the CUDA error code and can be added to exceptions
typedef boost::error_info<struct tag_cuda_err_code, int> cuda_err_code;

/// holds the CUDA error string and can be added to exceptions
typedef boost::error_info<struct tag_cuda_err_text, std::string> cuda_err_text;

/**
 * @brief device exception, thrown in case of a device related error
 *
 * @ingroup backend
 */
struct device_exception : virtual boost::exception, virtual std::exception
{

  /**
   * @brief Return a well formated string containing exception description
   */
  virtual const char* what() const throw()
  {
    std::ostringstream msg;
    msg << "CUDA error:\n";
    if(std::string const * mi = boost::get_error_info<cuda_err_text>(*this))
    {
      msg << "Error: " << *mi;
    }
    if(int const * mi = boost::get_error_info<cuda_err_code>(*this))
    {
      msg << " (Code: " << *mi << ")\n";
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

} // namespace cuda

} // namespace backend_detail

} // namespace mgpu


#endif // MGPU_BACKEND_CUDA_DEV_EXCEPTION_HPP
