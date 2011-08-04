// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef MGPU_INVOKE_HPP
#define MGPU_INVOKE_HPP

/**
 * @file invoke.hpp
 *
 * This header contains free functions to execute code in the context of the
 * different threads the mgpu environment provides.
 */

/**
 * @defgroup invoke
 * @ingroup main
 *
 * All invoke functions take a variable number of arguments up to
 * @c MGPU_DISP_MAX_PARAM_ARITY of any type:
 *
 * @code
 * template<typename func, typename T0, typename T1, ... typename Tn >
 * void invoke(func f, T0 p0, T1 p1, ... Tn pn, unsigned int thread_id);
 * @endcode
 *
 * and
 *
 * @code
 * template<typename func, typename T0, typename T1, ... typename Tn >
 * void invoke_all(func f, T0 p0, T1 p1, ... Tn pn);
 * @endcode
 */

#include <boost/bind.hpp>

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/seq/seq.hpp>

#include <mgpu/core/dev_rank.hpp>
#include <mgpu/environment.hpp>

namespace mgpu
{

/// forward-declare environment class
class environment;

/// generate the template arguments for the disp functions
#define MGPU_GENERATE_DISP_TEMPLATE_ARGS(z, n, _) , typename T ## n

/// generate the function arguments for the disp functions
#define MGPU_GENERATE_DISP_FUNCTION_ARGS(z, n, _) , T ## n p ## n

/// generate the boost::bind arguments for the disp functions
#define MGPU_GENERATE_DISP_BIND_ARGS(z, n, _) , p ## n

/**
 * @brief generate dispall functions
 *
 * The functions generated here allow users to add methods to the different
 * execution queues of the mgpu environment threads. Generates functions like:
 *
 * template<typename func, typename T0, typename T1, ... typename Tn >
 * inline void disp_gpu(func f, T0 p0, T1 p1, ... Tn pn)
 * {
 *   env::disp_dev(boost::bind(f, p0, p1, ... pn));
 * }
 *
 * @internal
 */
#define MGPU_GENERATE_DISP_ALL(z, n, method)                                   \
  template<typename func                                                       \
    BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GENERATE_DISP_TEMPLATE_ARGS, _)\
  >                                                                            \
  inline void BOOST_PP_SEQ_HEAD(method) (func f                                \
    BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GENERATE_DISP_FUNCTION_ARGS, _)\
  )                                                                            \
  {                                                                            \
    BOOST_PP_SEQ_TAIL(method)(                                                 \
      boost::bind(f                                                            \
        BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GENERATE_DISP_BIND_ARGS, _)\
      )                                                                        \
    );                                                                         \
  }                                                                            \
  /**/

/**
 * @brief generate disp functions
 *
 * The functions generated here allow users to add methods to the different
 * execution queues of the mgpu environment threads. Generates functions like:
 *
 * template<typename func, typename T0, typename T1, ... typename Tn >
 * inline void disp_gpu(func f, T0 p0, T1 p1, ... Tn pn, unsigned int thread_id)
 * {
 *   env::disp_dev(boost::bind(f, p0, p1, ... pn), thread_id);
 * }
 *
 * @internal
 */
#define MGPU_GENERATE_DISP_ONE(z, n, method)                                   \
  template<typename func                                                       \
    BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GENERATE_DISP_TEMPLATE_ARGS, _)\
  >                                                                            \
  inline void BOOST_PP_SEQ_HEAD(method) (func f                                \
    BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GENERATE_DISP_FUNCTION_ARGS, _)\
  , const dev_rank_t & rank)                                                   \
  {                                                                            \
  BOOST_PP_SEQ_TAIL(method)(                                                   \
    boost::bind(f                                                              \
      BOOST_PP_REPEAT_ ## z(BOOST_PP_INC(n), MGPU_GENERATE_DISP_BIND_ARGS, _)  \
    )                                                                          \
  , rank);                                                                     \
  }                                                                            \
  /**/



BOOST_PP_REPEAT(MGPU_DISP_MAX_PARAM_ARITY, MGPU_GENERATE_DISP_ONE,
                (invoke) (detail::runtime::invoke_device));
BOOST_PP_REPEAT(MGPU_DISP_MAX_PARAM_ARITY, MGPU_GENERATE_DISP_ALL,
                (invoke_all) (detail::runtime::invoke_all_devices));

/**
 * @brief invoke a function in a device thread
 *
 * @tparam func type of functor to invoke
 * @param f functor that should be invokeed
 * @param device id of device that should execute @c f
 * @ingroup invoke
 */
template<typename func>
inline void invoke(func f, const dev_rank_t & rank)
{
  detail::runtime::invoke_device(f, rank);
}

/**
 * @brief invoke a function in all device threads
 *
 * @tparam func type of functor to invoke
 * @param f functor that should be invokeed
 * @ingroup invoke
 */
template<typename func>
inline void invoke_all(func f)
{
  detail::runtime::invoke_all_devices(f);
}


} // namespace mgpu



#endif // MGPU_INVOKE_HPP
