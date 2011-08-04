// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#ifndef STOPWATCH_HPP
#define STOPWATCH_HPP

#ifdef __cplusplus

extern "C" void sw_start(const char * name);
extern "C" void sw_stop(const char * name);
extern "C" void sw_print();
extern "C" void sw_print2();
extern "C" double sw_get_time(const char * id);

#else

void sw_start(const char * name);
void sw_stop(const char * name);
void sw_print();
void sw_print2();
double sw_get_time(const char * id);

#endif

#endif // STOPWATCH_HPP
