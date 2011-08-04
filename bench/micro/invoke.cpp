// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



/**
 * Invocation micro-benchmark
 */

#include <stopwatch.hpp>
#include <mgpu/environment.hpp>
#include <mgpu/backend/backend.hpp>
#include <mgpu/synchronization.hpp>
#include <mgpu/invoke.hpp>

using namespace mgpu;


void func1(float * result, float arg1)
{
  *result = arg1;
}

void func2(float * result, float arg1, float arg2)
{
  *result = arg1 + arg2;
}

void func3(float * result, float arg1, float arg2, float arg3)
{
  *result = arg1 + arg2 + arg3;
}

void func4(float * result, float arg1, float arg2, float arg3, float arg4)
{
  *result = arg1 + arg2 + arg3 + arg4;
}



int main(void)
{
  // config -----

  int iterations = 50000;

  // -----

  int devices = MGPU_NUM_DEVICES;
  if(devices < 4)
  {
    printf("this benchmark is designed to run on "
      "systems with at least 4 devices\n");
    return 0;
  }

  environment e(dev_group::from_to(0, devices));
  barrier();

  float result;
  float * result_ptr = &result;

  float arg1 = 1.0;
  float arg2 = 1.0;
  float arg3 = 1.0;
  float arg4 = 1.0;


  // 1 argument ________________________________________________________________

  // normal -----

  // warmup
  func1(result_ptr, arg1);
  func1(result_ptr, arg1);

  sw_start("normal_func1");
  for(int i=0; i<iterations; i++)
  {
    func1(result_ptr, arg1);
    func1(result_ptr, arg1);
    func1(result_ptr, arg1);
    func1(result_ptr, arg1);
  }
  sw_stop("normal_func1");



  // bind -----

  // warmup

  boost::bind(func1, result_ptr, arg1)();
  boost::bind(func1, result_ptr, arg1)();

  barrier();
  sw_start("bind_func1");
  for(int i=0; i<iterations; i++)
  {
    boost::bind(func1, result_ptr, arg1)();
    boost::bind(func1, result_ptr, arg1)();
    boost::bind(func1, result_ptr, arg1)();
    boost::bind(func1, result_ptr, arg1)();
  }
  barrier();
  sw_stop("bind_func1");



  // function -----

  // warmup

  boost::function<void (void)> f1 = boost::bind(func1, result_ptr, arg1);
  f1();
  boost::function<void (void)> f2 = boost::bind(func1, result_ptr, arg1);
  f2();

  barrier();
  sw_start("function_func1");
  for(int i=0; i<iterations; i++)
  {
    boost::function<void (void)> f1 = boost::bind(func1, result_ptr, arg1);
    f1();
    boost::function<void (void)> f2 = boost::bind(func1, result_ptr, arg1);
    f2();
    boost::function<void (void)> f3 = boost::bind(func1, result_ptr, arg1);
    f3();
    boost::function<void (void)> f4 = boost::bind(func1, result_ptr, arg1);
    f4();
  }
  barrier();
  sw_stop("function_func1");



  // function copy -----

  // warmup

  boost::function<void (void)> f1_c = boost::bind(func1, result_ptr, arg1);
  f1_c();
  boost::function<void (void)> f2_c = boost::bind(func1, result_ptr, arg1);
  f2_c();

  barrier();
  sw_start("function_copy_func1");
  for(int i=0; i<iterations; i++)
  {
    boost::function<void (void)> f;
    boost::function<void (void)> f1 = boost::bind(func1, result_ptr, arg1);
    f = f1;
    f();
    boost::function<void (void)> f2 = boost::bind(func1, result_ptr, arg1);
    f = f2;
    f();
    boost::function<void (void)> f3 = boost::bind(func1, result_ptr, arg1);
    f = f3;
    f();
    boost::function<void (void)> f4 = boost::bind(func1, result_ptr, arg1);
    f = f4;
    f();
  }
  barrier();
  sw_stop("function_copy_func1");



  // invoke -----

  // warmup

  invoke_all(func1, result_ptr, arg1);
  invoke_all(func1, result_ptr, arg1);

  barrier();
  sw_start("invoke_func1");
  for(int i=0; i<iterations; i++)
  {
    invoke_all(func1, result_ptr, arg1);
  }
  //barrier();
  sw_stop("invoke_func1");






  // 2 arguments _______________________________________________________________

  // normal -----

  // warmup
  func2(result_ptr, arg1, arg2);
  func2(result_ptr, arg1, arg2);

  sw_start("normal_func2");
  for(int i=0; i<iterations*4; i++)
  {
    func2(result_ptr, arg1, arg2);
  }
  sw_stop("normal_func2");



  // bind -----

  // warmup

  boost::bind(func2, result_ptr, arg1, arg2)();
  boost::bind(func2, result_ptr, arg1, arg2)();

  barrier();
  sw_start("bind_func2");
  for(int i=0; i<iterations; i++)
  {
    boost::bind(func2, result_ptr, arg1, arg2)();
    boost::bind(func2, result_ptr, arg1, arg2)();
    boost::bind(func2, result_ptr, arg1, arg2)();
    boost::bind(func2, result_ptr, arg1, arg2)();
  }
  barrier();
  sw_stop("bind_func2");



  // function -----

  // warmup

  boost::function<void (void)> f1_2 =
    boost::bind(func2, result_ptr, arg1, arg2);
  f1_2();
  boost::function<void (void)> f2_2 =
    boost::bind(func2, result_ptr, arg1, arg2);
  f2_2();

  barrier();
  sw_start("function_func2");
  for(int i=0; i<iterations; i++)
  {
    boost::function<void (void)> f1 =
      boost::bind(func2, result_ptr, arg1, arg2);
    f1();
    boost::function<void (void)> f2 =
      boost::bind(func2, result_ptr, arg1, arg2);
    f2();
    boost::function<void (void)> f3 =
      boost::bind(func2, result_ptr, arg1, arg2);
    f3();
    boost::function<void (void)> f4 =
      boost::bind(func2, result_ptr, arg1, arg2);
    f4();
  }
  barrier();
  sw_stop("function_func2");



  // function copy -----

  // warmup

  boost::function<void (void)> f1_2_c =
    boost::bind(func2, result_ptr, arg1, arg2);
  f1_2_c();
  boost::function<void (void)> f2_2_c =
    boost::bind(func2, result_ptr, arg1, arg2);
  f2_2_c();

  barrier();
  sw_start("function_copy_func2");
  for(int i=0; i<iterations; i++)
  {
    boost::function<void (void)> f;
    boost::function<void (void)> f1 =
      boost::bind(func2, result_ptr, arg1, arg2);
    f = f1;
    f();
    boost::function<void (void)> f2 =
      boost::bind(func2, result_ptr, arg1, arg2);
    f = f2;
    f();
    boost::function<void (void)> f3 =
      boost::bind(func2, result_ptr, arg1, arg2);
    f = f3;
    f();
    boost::function<void (void)> f4 =
      boost::bind(func2, result_ptr, arg1, arg2);
    f = f4;
    f();
  }
  barrier();
  sw_stop("function_copy_func2");



  // invoke -----

  // warmup

  invoke_all(func2, result_ptr, arg1, arg2);
  invoke_all(func2, result_ptr, arg1, arg2);

  barrier();
  sw_start("invoke_func2");
  for(int i=0; i<iterations; i++)
  {
    invoke_all(func2, result_ptr, arg1, arg2);
  }
  barrier();
  sw_stop("invoke_func2");



  // 3 arguments _______________________________________________________________

  // normal -----

  // warmup
  func3(result_ptr, arg1, arg2, arg3);
  func3(result_ptr, arg1, arg2, arg3);

  sw_start("normal_func3");
  for(int i=0; i<iterations*4; i++)
  {
    func3(result_ptr, arg1, arg2, arg3);
  }
  sw_stop("normal_func3");


  // bind -----

  // warmup

  boost::bind(func3, result_ptr, arg1,arg2, arg3)();
  boost::bind(func3, result_ptr, arg1,arg2, arg3)();

  barrier();
  sw_start("bind_func3");
  for(int i=0; i<iterations; i++)
  {
    boost::bind(func3, result_ptr, arg1, arg2, arg3)();
    boost::bind(func3, result_ptr, arg1, arg2, arg3)();
    boost::bind(func3, result_ptr, arg1, arg2, arg3)();
    boost::bind(func3, result_ptr, arg1, arg2, arg3)();
  }
  barrier();
  sw_stop("bind_func3");



  // function -----

  // warmup

  boost::function<void (void)> f1_3 = boost::bind(func3, result_ptr,
    arg1, arg2, arg3);
  f1_3();
  boost::function<void (void)> f2_3 = boost::bind(func3, result_ptr,
    arg1, arg2, arg3);
  f2_3();

  barrier();
  sw_start("function_func3");
  for(int i=0; i<iterations; i++)
  {
    boost::function<void (void)> f1 = boost::bind(func3, result_ptr,
      arg1, arg2, arg3);
    f1();
    boost::function<void (void)> f2 = boost::bind(func3, result_ptr,
      arg1, arg2, arg3);
    f2();
    boost::function<void (void)> f3 = boost::bind(func3, result_ptr,
      arg1, arg2, arg3);
    f3();
    boost::function<void (void)> f4 = boost::bind(func3, result_ptr,
      arg1, arg2, arg3);
    f4();
  }
  barrier();
  sw_stop("function_func3");



  // function copy -----

  // warmup

  boost::function<void (void)> f1_3_c = boost::bind(func3, result_ptr,
    arg1, arg2, arg3);
  f1_3_c();
  boost::function<void (void)> f2_3_c = boost::bind(func3, result_ptr,
    arg1, arg2, arg3);
  f2_3_c();

  barrier();
  sw_start("function_copy_func3");
  for(int i=0; i<iterations; i++)
  {
    boost::function<void (void)> f;
    boost::function<void (void)> f1 = boost::bind(func3, result_ptr,
      arg1, arg2, arg3);
    f = f1;
    f();
    boost::function<void (void)> f2 = boost::bind(func3, result_ptr,
      arg1, arg2, arg3);
    f = f2;
    f();
    boost::function<void (void)> f3 = boost::bind(func3, result_ptr,
      arg1, arg2, arg3);
    f = f3;
    f();
    boost::function<void (void)> f4 = boost::bind(func3, result_ptr,
      arg1, arg2, arg3);
    f = f4;
    f();
  }
  barrier();
  sw_stop("function_copy_func3");


  // invoke -----

  // warmup

  invoke_all(func3, result_ptr, arg1, arg2, arg3);
  invoke_all(func3, result_ptr, arg1, arg2, arg3);

  barrier();
  sw_start("invoke_func3");
  for(int i=0; i<iterations; i++)
  {
    invoke_all(func3, result_ptr, arg1, arg2, arg3);
  }
  barrier();
  sw_stop("invoke_func3");



  // 4 arguments _______________________________________________________________

  // normal -----

  // warmup
  func4(result_ptr, arg1, arg2, arg3, arg4);
  func4(result_ptr, arg1, arg2, arg3, arg4);

  sw_start("normal_func4");
  for(int i=0; i<iterations*4; i++)
  {
    func4(result_ptr, arg1, arg2, arg3, arg4);
  }
  sw_stop("normal_func4");


  // bind -----

  // warmup

  boost::bind(func4, result_ptr, arg1,arg2, arg4, arg4)();
  boost::bind(func4, result_ptr, arg1,arg2, arg4, arg4)();

  barrier();
  sw_start("bind_func4");
  for(int i=0; i<iterations; i++)
  {
    boost::bind(func4, result_ptr, arg1, arg2, arg4, arg4)();
    boost::bind(func4, result_ptr, arg1, arg2, arg4, arg4)();
    boost::bind(func4, result_ptr, arg1, arg2, arg4, arg4)();
    boost::bind(func4, result_ptr, arg1, arg2, arg4, arg4)();
  }
  barrier();
  sw_stop("bind_func4");



  // function -----

  // warmup

  boost::function<void (void)> f1_4 = boost::bind(func4, result_ptr,
    arg1, arg2, arg4, arg4);
  f1_4();
  boost::function<void (void)> f2_4 = boost::bind(func4, result_ptr,
    arg1, arg2, arg4, arg4);
  f2_4();

  barrier();
  sw_start("function_func4");
  for(int i=0; i<iterations; i++)
  {
    boost::function<void (void)> f1 = boost::bind(func4, result_ptr,
      arg1, arg2, arg4, arg4);
    f1();
    boost::function<void (void)> f2 = boost::bind(func4, result_ptr,
      arg1, arg2, arg4, arg4);
    f2();
    boost::function<void (void)> f3 = boost::bind(func4, result_ptr,
      arg1, arg2, arg4, arg4);
    f3();
    boost::function<void (void)> f4 = boost::bind(func4, result_ptr,
      arg1, arg2, arg4, arg4);
    f4();
  }
  barrier();
  sw_stop("function_func4");



  // function copy -----

  // warmup

  boost::function<void (void)> f1_4_c = boost::bind(func4, result_ptr,
    arg1, arg2, arg4, arg4);
  f1_4_c();
  boost::function<void (void)> f2_4_c = boost::bind(func4, result_ptr,
    arg1, arg2, arg4, arg4);
  f2_4_c();

  barrier();
  sw_start("function_copy_func4");
  for(int i=0; i<iterations; i++)
  {
    boost::function<void (void)> f;
    boost::function<void (void)> f1 = boost::bind(func4, result_ptr,
      arg1, arg2, arg4, arg4);
    f = f1;
    f();
    boost::function<void (void)> f2 = boost::bind(func4, result_ptr,
      arg1, arg2, arg4, arg4);
    f = f2;
    f();
    boost::function<void (void)> f3 = boost::bind(func4, result_ptr,
      arg1, arg2, arg4, arg4);
    f = f3;
    f();
    boost::function<void (void)> f4 = boost::bind(func4, result_ptr,
      arg1, arg2, arg4, arg4);
    f = f4;
    f();
  }
  barrier();
  sw_stop("function_copy_func4");


  // invoke -----

  // warmup

  invoke_all(func4, result_ptr, arg1, arg2, arg3, arg4);
  invoke_all(func4, result_ptr, arg1, arg2, arg3, arg4);

  barrier();
  sw_start("invoke_func4");
  for(int i=0; i<iterations; i++)
  {
    invoke_all(func4, result_ptr, arg1, arg2, arg3, arg4);
  }
  barrier();
  sw_stop("invoke_func4");




  sw_print2();
}
