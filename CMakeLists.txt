# -----
# Copyright 2011 Sebastian Schaetz. Distributed under the Boost
# Software License, Version 1.0. (See accompanying file
# LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# -----


cmake_minimum_required (VERSION 2.8)


# external dependencies -----

# set(CUDA_TOOLKIT_ROOT_DIR "YOUR/CUDA/PATH")
# set(BOOST_ROOT "YOUR/BOOST/PATH")

# ignore this
#IF(UNIX)
#    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda40/")
#ENDIF()

#IF(WIN32)
#    set(CUDA_TOOLKIT_ROOT_DIR 
#        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0/")
#    set(BOOST_ROOT "C:/dev/lib/boost_1_45_0/")
#ENDIF()

find_package(CUDA)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_ADDITIONAL_VERSIONS "1.47" "1.47.0")
find_package(Boost 1.45.0 COMPONENTS thread unit_test_framework)


# get system information -----

# number of devices
TRY_RUN(MGPU_NUM_DEVICES COMPILE_RESULT_VAR
        ${CMAKE_CURRENT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/tools/cuda/num_devices.cpp
        CMAKE_FLAGS 
            -DLINK_LIBRARIES:STRING=${CUDA_LIBRARIES}
            -DINCLUDE_DIRECTORIES:STRING=${CUDA_INCLUDE_DIRS}
       )

# information about P2P capabilities
TRY_RUN(RUN_RESULT_VAR COMPILE_RESULT_VAR
        ${CMAKE_CURRENT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/tools/cuda/p2p_capabilities.cpp
        CMAKE_FLAGS 
            -DLINK_LIBRARIES:STRING=${CUDA_LIBRARIES}
            -DINCLUDE_DIRECTORIES:STRING=${CUDA_INCLUDE_DIRS}
        RUN_OUTPUT_VARIABLE MGPU_P2P_CAPABILITIES
       )
       
# information about compute capabilities
TRY_RUN(RUN_RESULT_VAR COMPILE_RESULT_VAR
        ${CMAKE_CURRENT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/tools/cuda/compute_capabilities.cpp
        CMAKE_FLAGS 
            -DLINK_LIBRARIES:STRING=${CUDA_LIBRARIES}
            -DINCLUDE_DIRECTORIES:STRING=${CUDA_INCLUDE_DIRS}
        RUN_OUTPUT_VARIABLE MGPU_COMPUTE_CAPABILITIES
       )


#configuration -----

message("Detected ${MGPU_NUM_DEVICES} CUDA devices")
message("Generating code for compute capabilities ${MGPU_COMPUTE_CAPABILITIES}")

set(MGPU_VERSION 0.1)
set(MGPU_RUNTIME_QUEUE_TYPE queue_lockfree)
set(MGPU_BACKEND cuda)
string(TOUPPER ${MGPU_BACKEND} MGPU_BACKEND_UPPER)

configure_file(
  "${PROJECT_SOURCE_DIR}/include/mgpu/config.hpp.in"
  "${PROJECT_SOURCE_DIR}/include/mgpu/config.hpp"
)


# compiler flags -----

IF(UNIX)
SET(CMAKE_CXX_FLAGS "-Wall -O3 -g")
ENDIF()

IF(WIN32)
# disable warning 4514 and warning 4530 and warning 4996
SET(CMAKE_CXX_FLAGS "/wd4514 /EHsc /D_SCL_SECURE_NO_WARNINGS")
ENDIF()


SET(CUDA_NVCC_FLAGS "-O3")
FOREACH(f ${MGPU_COMPUTE_CAPABILITIES})
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    "--gpu-architecture=compute_${f}" "--gpu-code=sm_${f}")
ENDFOREACH()


# directories -----

include_directories("${PROJECT_SOURCE_DIR}/include/")
include_directories(${Boost_INCLUDE_DIRS})

link_directories(${Boost_LIBRARY_DIRS})


# library -----

FILE(GLOB_RECURSE MGPU_LIB_SOURCES1 ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
FILE(GLOB_RECURSE MGPU_LIB_SOURCES2 ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cu)
CUDA_ADD_LIBRARY(mgpu "${MGPU_LIB_SOURCES1};${MGPU_LIB_SOURCES2}" STATIC)


# unit tests -----

enable_testing()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/unit/)


# benchmarks -----

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/bench/)


# examples -----

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/example/)

