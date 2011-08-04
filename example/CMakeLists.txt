# -----
# Copyright 2011 Sebastian Schaetz. Distributed under the Boost
# Software License, Version 1.0. (See accompanying file
# LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# -----


cmake_minimum_required (VERSION 2.8)


MACRO(MGPU_REGISTER_EXAMPLE examplename file0)

    SET(SOURCE_FILES ${file0})
  if(${ARGC} GREATER 2)
    foreach(f ${ARGN})
      SET(SOURCE_FILES ${SOURCE_FILES} " " ${f})
    endforeach()
  endif()
  
  SET(binaryname "example.${examplename}")
  
  CUDA_ADD_EXECUTABLE(${binaryname} ${SOURCE_FILES})
  target_link_libraries(${binaryname} mgpu
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${Boost_THREAD_LIBRARY})
  add_test(${examplename} ${binaryname})
ENDMACRO()


MACRO(MGPU_REGISTER_EXAMPLE_FFT examplename file0)

    SET(SOURCE_FILES ${file0})
  if(${ARGC} GREATER 2)
    foreach(f ${ARGN})
      SET(SOURCE_FILES ${SOURCE_FILES} " " ${f})
    endforeach()
  endif()
  
  SET(binaryname "example.${examplename}")
  
  CUDA_ADD_EXECUTABLE(${binaryname} ${SOURCE_FILES})
  target_link_libraries(${binaryname} mgpu
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${Boost_THREAD_LIBRARY})
  CUDA_ADD_CUFFT_TO_TARGET(${binaryname})
  add_test(${examplename} ${binaryname})
ENDMACRO()


MGPU_REGISTER_EXAMPLE(broadcast broadcast.cpp)
MGPU_REGISTER_EXAMPLE(environment environment.cpp)
MGPU_REGISTER_EXAMPLE_FFT(fft fft.cpp)
MGPU_REGISTER_EXAMPLE(invoke_kernel invoke_kernel.cpp)
MGPU_REGISTER_EXAMPLE(simple_seg_dev_vector simple_seg_dev_vector.cpp)
MGPU_REGISTER_EXAMPLE(simple_transfer simple_transfer.cpp)
