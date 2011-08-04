# -----
# Copyright 2011 Sebastian Schaetz. Distributed under the Boost
# Software License, Version 1.0. (See accompanying file
# LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# -----


cmake_minimum_required (VERSION 2.8)


MACRO(MGPU_REGISTER_BENCHMARK benchname file0)
  
  SET(SOURCE_FILES ${file0} "../stopwatch.cpp")
  if(${ARGC} GREATER 2)
    foreach(f ${ARGN})
      SET(SOURCE_FILES ${SOURCE_FILES} " " ${f})
    endforeach()
  endif()
  
  SET(binaryname "bench.${benchname}")
  
  CUDA_ADD_EXECUTABLE(${binaryname} ${SOURCE_FILES})
  target_link_libraries(${binaryname} mgpu
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${Boost_THREAD_LIBRARY})
  add_test(${benchname} ${binaryname})
ENDMACRO()


MACRO(MGPU_REGISTER_BENCHMARK_CUBLAS benchname file0)
  
  SET(SOURCE_FILES ${file0} "../stopwatch.cpp")
  if(${ARGC} GREATER 2)
    foreach(f ${ARGN})
      SET(SOURCE_FILES ${SOURCE_FILES} " " ${f})
    endforeach()
  endif()
  
  SET(binaryname "bench.${benchname}")
  
  CUDA_ADD_EXECUTABLE(${binaryname} ${SOURCE_FILES})
  target_link_libraries(${binaryname} mgpu
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${Boost_THREAD_LIBRARY})
  CUDA_ADD_CUBLAS_TO_TARGET(${binaryname})
  add_test(${benchname} ${binaryname})
ENDMACRO()


MACRO(MGPU_REGISTER_BENCHMARK_CUFFT benchname file0)
  
  SET(SOURCE_FILES ${file0} "../stopwatch.cpp")
  if(${ARGC} GREATER 2)
    foreach(f ${ARGN})
      SET(SOURCE_FILES ${SOURCE_FILES} " " ${f})
    endforeach()
  endif()
  
  SET(binaryname "bench.${benchname}")
  
  CUDA_ADD_EXECUTABLE(${binaryname} ${SOURCE_FILES})
  target_link_libraries(${binaryname} mgpu
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${Boost_THREAD_LIBRARY})
  CUDA_ADD_CUFFT_TO_TARGET(${binaryname})
  add_test(${benchname} ${binaryname})
ENDMACRO()


include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/micro)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/algorithm)

