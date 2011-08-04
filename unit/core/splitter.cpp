// -----
// Copyright 2011 Sebastian Schaetz. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
// -----



#define BOOST_TEST_MODULE core.splitter

#include <boost/test/unit_test.hpp>

#include <mgpu/core/detail/splitter.hpp>


using namespace mgpu::detail;



// exception
//______________________________________________________________________________

BOOST_AUTO_TEST_CASE(exception)
{
  BOOST_REQUIRE_THROW(splitter s(127, 8, 4), mgpu::mgpu_exception);
  BOOST_REQUIRE_THROW(splitter s(128, 8, 0), mgpu::mgpu_exception);
}

// functionality
//______________________________________________________________________________

BOOST_AUTO_TEST_CASE(functionality)
{
  {
    int size = 128; int blocksize = 8; int vectors = 4;
    int result[] = {32, 32, 32, 32, 0, 0, 0, 0};

    splitter s(size, blocksize, vectors);
    for(unsigned int i=0; i<sizeof(result)/sizeof(int); i++)
    {
      BOOST_CHECK_EQUAL(result[i], s++);
    }
  }

  {
    int size = 128; int blocksize = 8; int vectors = 6;
    int result[] = {24, 24, 24, 24, 16, 16, 0, 0};

    splitter s(size, blocksize, vectors);
    for(unsigned int i=0; i<sizeof(result)/sizeof(int); i++)
    {
      BOOST_CHECK_EQUAL(result[i], s++);
    }
  }

  {
    int size = 32; int blocksize = 8; int vectors = 6;
    int result[] = {8, 8, 8, 8, 0, 0, 0, 0};

    splitter s(size, blocksize, vectors);
    for(unsigned int i=0; i<sizeof(result)/sizeof(int); i++)
    {
      BOOST_CHECK_EQUAL(result[i], s++);
    }
  }

  {
    int size = 32; int blocksize = 8; int vectors = 1;
    int result[] = {32, 0, 0, 0, 0, 0, 0, 0};

    splitter s(size, blocksize, vectors);
    for(unsigned int i=0; i<sizeof(result)/sizeof(int); i++)
    {
      BOOST_CHECK_EQUAL(result[i], s++);
    }
  }
}
