// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#include <vector>
#include <stdio.h>
#include <map>
#include <string.h>

#if defined(__GNUC__)
#include <time.h>
#elif defined(_MSC_VER)
#include <Windows.h>
#endif


#include "stopwatch.hpp"

// all the nasty stuff is hidden in this namespace
namespace stopwatch
{
#if defined(_MSC_VER)
  inline double get_frequency()
  {
    LARGE_INTEGER proc_freq;
    ::QueryPerformanceFrequency(&proc_freq);
    return 1000.*1000.*1000./(double)proc_freq.QuadPart;
  }
  static const double frequency = get_frequency();
#endif

  unsigned int depth = 0;
  unsigned int counter = 0;


  // function to get current time
  inline double current_time()
  {
#if defined(__GNUC__)
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double t = ts.tv_sec * 1000 * 1000 * 1000;
    t += ts.tv_nsec;
    return t;
#elif defined(_MSC_VER)
    LARGE_INTEGER ts;
    ::QueryPerformanceCounter(&ts);
    return (double)ts.QuadPart/(double)frequency;
#endif
  }
  
  struct stopwatch_object
  {
    stopwatch_object() : name_(0), depth_(0), timestamp_(0.0) {}
    stopwatch_object(const char *& name) : 
    name_(name), depth_(depth), timestamp_(current_time()) 
    {}
    
    const char * name_;
    unsigned int depth_;
    double timestamp_;
  };
  
  struct map_entry
  {
    stopwatch::stopwatch_object swo;
    bool even;
    unsigned int invocations;
    double timestamp_buff;
    unsigned int num;
  };
  
  // this initialization may cause the stopwatch to segfault if more than
  // data.size()/2 different measurements are taken
  std::vector<stopwatch_object> data(500000);
}

// return the time differente between the first and the second occurence of id
double sw_get_time(const char * id)
{
  double result = 0.0;
  double tmp = 0.0;
  for(unsigned int i=0; i<stopwatch::data.size(); i++)
  {
    if(stopwatch::data[i].timestamp_ == 0.0)
    {
      return result;
    }

    if(0 == strcmp(id, stopwatch::data[i].name_))
    {
      if(tmp == 0.0)
      {
        tmp = stopwatch::data[i].timestamp_;
      }
      else
      {
        result += (stopwatch::data[i].timestamp_ - tmp);
        tmp = 0.0;
      }
    }

  }
  return result;
}


// start the stopwatch
void sw_start(const char * name)
{
  stopwatch::depth++;
  stopwatch::data[stopwatch::counter] = stopwatch::stopwatch_object(name);
  stopwatch::counter++;
} 

// stop the stopwatch
void sw_stop(const char * name)
{
  stopwatch::data[stopwatch::counter] = stopwatch::stopwatch_object(name);
  stopwatch::depth--;
  stopwatch::counter++;
} 

// print contents of entire stopwatch
void sw_print()
{
  for(unsigned int i=0; i<stopwatch::data.size(); i++)
  {
    if(stopwatch::data[i].timestamp_ == 0.0)
    {
      printf("%d entries\n", i);
      return;
    }
    for(unsigned short int j=0; j<stopwatch::data[i].depth_-1; j++)
    {
      printf("\t");
    }
    printf("%s %fms\n", stopwatch::data[i].name_, 
      stopwatch::data[i].timestamp_/1000/1000);
  }
}

// process and print contents of entire stopwatch
void sw_print2()
{

  typedef std::map<const char *, stopwatch::map_entry> data;
  typedef std::pair<const char *, stopwatch::map_entry> dataentry;
  typedef data::iterator it_type;
  
  data mymap;
  unsigned int num = 0;
  for(unsigned int i=0; i<stopwatch::data.size(); i++)
  {
    if(stopwatch::data[i].timestamp_ == 0.0)
    {
      break;
    }
    
    data::iterator it = mymap.find(stopwatch::data[i].name_);
    
    if(it == mymap.end())
    {
      stopwatch::map_entry e;
      e.swo = stopwatch::data[i];
      e.even = false;
      e.invocations = 1;
      e.swo.timestamp_ = 0;
      e.timestamp_buff = stopwatch::data[i].timestamp_;
      e.num = num;
      num++;
      mymap.insert(dataentry(stopwatch::data[i].name_, e));
    }
    else
    {
      if(!it->second.even)
      {
        it->second.swo.timestamp_ += (stopwatch::data[i].timestamp_ -
          it->second.timestamp_buff);
        it->second.even = true;
      }
      else
      {
        it->second.timestamp_buff = stopwatch::data[i].timestamp_;
        it->second.even = false;
        it->second.invocations++;
      }
    }
  }

  data::iterator it;
  for(unsigned int i=0; i<num; i++)
  {
    for(it = mymap.begin(); it!=mymap.end(); ++it)
    {
      if(i == it->second.num)
      {
        for(unsigned int i=0; i<it->second.swo.depth_; i++)
        {
          printf("\t");
        }
        printf("%s (%d) %fms\n", it->second.swo.name_, 
          it->second.invocations, it->second.swo.timestamp_/1000/1000);
      }
    }
  }
}
