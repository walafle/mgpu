// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_INVOKE_KERNEL_HPP
#define MGPU_INVOKE_KERNEL_HPP

/**
 * @file invoke_kernel.hpp
 *
 * This header contains free functions to execute CUDA kernels.
 */

/**
 * @defgroup kernelinvoke Kernel Dispatch
 * @ingroup main
 *
 * All kernel invoke functions take a variable number of arguments up to
 * @c MGPU_DISP_MAX_PARAM_ARITY of any type:
 *
 * @code
 * template<typename func, typename T0, typename T1, ... typename Tn >
 * void disp_kernel(func f, T0 p0, T1 p1, ... Tn pn, unsigned int device_id);
 * @endcode
 *
 * and
 *
 * @code
 * template<typename func, typename T0, typename T1, ... typename Tn >
 * void dispall_kernel(func f, T0 p0, T1 p1, ... Tn pn);
 * @endcode
 *
 * If a distributed vector is passed to a kernel caller, the kernel caller will
 * receive the @c dev_vector that is locatet on the current device instead of
 * the @c dev_dist_vector.
 */

#include <boost/bind.hpp>


#include <mgpu/invoke.hpp>
#include <mgpu/core/dev_rank.hpp>
#include <mgpu/container/seg_dev_vector.hpp>
#include <mgpu/seg_dev_stream.hpp>

namespace mgpu
{

/// the pass_through class resembles effectively a simulated reference
/// it's type however has a special meaning in the context of invoking kernels
template<class T>
struct pass_through
{
  public:

    typedef T type;

    pass_through(): t_(NULL) {}
    explicit pass_through(T& t): t_(&t) {}
    explicit pass_through(const T& t): t_(const_cast<T*>(&t)) {}

    operator T& () const { return *t_; }
    T& get() const { return *t_; }
    T* get_pointer() const { return t_; }

    void set(T & t) { *t_ = t; }
    void set(const T & t) { *t_ = const_cast<T>(t); }

  private:

    T* t_;
};

namespace detail
{

struct pass_dev_rank_ {};

struct pass_dev_id_ {};

struct select_one_ {};

/**
 * @brief wrapper to have one call for built-in types and special types
 */
template<typename T>
T & invoke_kernel_impl_arg_wrapper(T * data, const int &)
{
  return *data;
}

/// if a seg_dev_vector is passed, we pass a device range to each function
template<typename T, typename Alloc>
dev_range<T> invoke_kernel_impl_arg_wrapper(
  seg_dev_vector<T, Alloc> * vec, const int & segment)
{
  return vec->local(segment);
}

/// if a dev_stream is passes, we pass the local stream to each function
inline backend::dev_stream invoke_kernel_impl_arg_wrapper(
  seg_dev_stream * stream, const int & segment)
{
  return (*stream)[segment];
}

/// if a seg_dev_iterator is passed, we pass a local iterator to each function
template<typename T, typename Alloc>
dev_iterator<T> invoke_kernel_impl_arg_wrapper(
  seg_dev_iterator<T, Alloc> * iter, const int & segment)
{
  return iter->begin_local(segment);
}

/// if a pass_through<T> is passed, we pass the object through unmodified
template<typename T>
T & invoke_kernel_impl_arg_wrapper(pass_through<T> * p, const int &)
{
  return p->get();
}

/// if a pass_dev_rank_ is passed, the device rank is passed through
inline dev_rank_t invoke_kernel_impl_arg_wrapper(const pass_dev_rank_ * r,
  const int & rank)
{
  return rank;
}

/// if a pass_dev_id_ is passed, the device id is passed through
inline dev_id_t invoke_kernel_impl_arg_wrapper(const pass_dev_id_ * r,
  const int & rank)
{
  return environment::dev_id(rank);
}

/// if a select_one_ is passed, one device is passed true, the others false
inline bool invoke_kernel_impl_arg_wrapper(const select_one_ * r,
  const int & rank)
{
  return (rank == 0) ? true : false;
}

} // namespace detail

/// pass the device id to the function
extern const detail::pass_dev_id_ pass_dev_id;

/// pass the device rank to the function
extern const detail::pass_dev_rank_ pass_dev_rank;

/// pass true to one of the functions
extern const detail::select_one_ select_one;


// generate template argument list for declarations
#define MGPU_GEN_DISP_K_TEMPLATE_ARGS_TN(z, n, _) , typename T ## n

// generate argument list for declarations
#define MGPU_GEN_DISP_K_TEMPLATE_ARGS(z, n, _) , T ## n

// generate argument list for invoke_kernel_impl (takes pointers)
#define MGPU_GEN_DISP_K_FUNCTION_PTR_ARGS(z, n, _) , T ## n * p ## n

// generate argument list for the user interface
#define MGPU_GEN_DISP_K_FUNCTION_REF_ARGS(z, n, _) , T ## n & p ## n

// wrap arguments and pass to actual function
#define MGPU_GEN_DISP_K_CALLER_ARGS(z, n, _) BOOST_PP_COMMA_IF(n)              \
  detail::invoke_kernel_impl_arg_wrapper(p ## n, rank)

// generate the arguments for invoke_kernel_impl with addresses
#define MGPU_GEN_DISP_K_CALLER_ARGS_REF(z, n, _) , &p ## n


/**
 * @brief generate dispall_kernel functions
 *
 * The functions generated here allow users to call GPU kernels with distributed
 * vectors. The macro generates functions take any number of arguments and a
 * kernel caller function.
 *
 * The kernel caller function is called in each device thread. The arguments
 * are passed through with the exception of distributed vectors. If they are
 * passed they are handled in a special way. A distributed vector is passed as a
 * device vector, each kernel caller function gets passed a reference to the
 * part of the distributed vector that is located on the current device.
 *
 * @internal
 */
#define MGPU_GENERATE_DISP_ALL_KERNEL(z, n, _)                                 \
namespace detail                                                               \
{                                                                              \
 template <typename func                                                       \
   BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GEN_DISP_K_TEMPLATE_ARGS_TN, _) \
 >                                                                             \
 void invoke_kernel_impl(func kernel_caller                                    \
   BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GEN_DISP_K_FUNCTION_PTR_ARGS, _)\
 , const dev_rank_t & rank)                                                    \
 {                                                                             \
   kernel_caller(                                                              \
     BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GEN_DISP_K_CALLER_ARGS, _)    \
   );                                                                          \
 }                                                                             \
}                                                                              \
template<typename func                                                         \
  BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GEN_DISP_K_TEMPLATE_ARGS_TN, _)  \
>                                                                              \
void invoke_kernel_all(func kernel_caller                                      \
 BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GEN_DISP_K_FUNCTION_REF_ARGS, _)  \
)                                                                              \
{                                                                              \
 for(dev_rank_t rank=0; (unsigned)rank<detail::runtime::size(); rank++)        \
 {                                                                             \
   invoke(detail::invoke_kernel_impl<func                                      \
     BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GEN_DISP_K_TEMPLATE_ARGS, _)  \
   >, kernel_caller                                                            \
     BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GEN_DISP_K_CALLER_ARGS_REF, _)\
   , rank, rank);                                                              \
 }                                                                             \
}                                                                              \
/**/

/**
 * @brief generate disp_kernel functions
 *
 * Similar to the generation of dispall_kernel functions with the exception that
 * the kernel caller is only called once for the specified device id.
 *
 * Distributed vectors are translated to local device vectors in the same way as
 * in the dispall_kernel functions case.
 *
 * @internal
 */
#define MGPU_GENERATE_DISP_ONE_KERNEL(z, n, _)                                 \
template<typename func                                                         \
  BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GEN_DISP_K_TEMPLATE_ARGS_TN, _)  \
>                                                                              \
void invoke_kernel(func kernel_caller                                          \
  BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GEN_DISP_K_FUNCTION_REF_ARGS, _) \
  , const dev_rank_t & rank)                                                   \
{                                                                              \
   invoke(detail::invoke_kernel_impl<func                                      \
     BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GEN_DISP_K_TEMPLATE_ARGS, _)  \
   >, kernel_caller                                                            \
     BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GEN_DISP_K_CALLER_ARGS_REF, _)\
   , rank, rank);                                                              \
}                                                                              \
/**/

BOOST_PP_REPEAT(MGPU_DISP_MAX_PARAM_ARITY, MGPU_GENERATE_DISP_ALL_KERNEL, _)
BOOST_PP_REPEAT(MGPU_DISP_MAX_PARAM_ARITY, MGPU_GENERATE_DISP_ONE_KERNEL, _)


/**
 * @brief invoke a kernel on a device
 *
 * @tparam func type of kernel calling functor to invoke
 * @param f kernel caller that should be invokeed
 * @param deviceid id of device that should execute @c f
 * @ingroup kernelinvoke
 */
template<typename func>
inline void invoke_kernel(func kernel_caller, const dev_rank_t & rank)
{
  invoke(kernel_caller, rank);
}

/**
 * @brief invoke a kernel on all devices
 *
 * @tparam func type of kernel calling functor to invoke
 * @param f kernel caller that should be invokeed
 * @ingroup kernelinvoke
 */
template<typename func>
inline void invoke_kernel_all(func kernel_caller)
{
  invoke_all(kernel_caller);
}


} // namespace mgpu



#endif // MGPU_INVOKE_KERNEL_HPP
