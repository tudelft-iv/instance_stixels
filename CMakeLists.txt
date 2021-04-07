# This file is part of Instance Stixels:
# https://github.com/tudelft-iv/instance-stixels
#
# Originally, it was part of stixels:
# https://github.com/dhernandez0/stixels

# Copyright (c) 2016 Daniel Hernandez Juarez.
# Copyright (c) 2019 Thomas Hehn.

# Instance Stixels is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Instance Stixels is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Instance Stixels. If not, see <http://www.gnu.org/licenses/>.

#cmake_minimum_required(VERSION 2.4)
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(TEST_SOURCES tests/main.cpp 
#    tests/segmentation_tests.cpp tests/stixelskernels_tests.cu)
#set(TESTED_SOURCES Segmentation.cpp StixelsKernels.cu)

# Find libraries
project(InstanceStixels)
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

# Optional, only for run_cityscapes
find_package( HDF5 COMPONENTS CXX )

# Optional, only for ros node
find_package( catkin COMPONENTS # not required
  roscpp
  sensor_msgs
  std_msgs
  visualization_msgs
  instance_stixels_msgs
  message_generation
  cv_bridge
  image_transport
  image_geometry
  dynamic_reconfigure
  )

# Set variables
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#~ # generate test data if nonexistent
#~ if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/testdata.h")
#~     message("${CMAKE_CURRENT_SOURCE_DIR}/tests/testdata.h not found. "
#~             "Executing python script.")
#~     execute_process(COMMAND python3 generate_testdata.py WORKING_DIRECTORY
#~                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/)
#~ endif()

# TensorRT. Adapted from:
# https://github.com/onnx/onnx-tensorrt/blob/master/CMakeLists.txt
#set(TENSORRT_ROOT "/home/thehn/Code/TensorRT/TensorRT-7.0.0.11")
if (NOT "$ENV{TRT_RELEASE}" STREQUAL "")
  set(TENSORRT_ROOT "$ENV{TRT_RELEASE}" CACHE
      INTERNAL "Copied from environment variable")
endif()
message("Tensorrt root: ${TENSORRT_ROOT}")
# headers
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
#find_path(TENSORRT_INCLUDE_DIR NvOnnxParser.h
#  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
#  PATH_SUFFIXES include)
#MESSAGE(STATUS "Found NvOnnxParser headers at ${TENSORRT_INCLUDE_DIR}")
# libs
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_MYELIN myelin
    HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY 
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_ONNXPARSER}
    ${TENSORRT_LIBRARY_MYELIN})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
    TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
    message(ERROR "Cannot find TensorRT library.")
endif()

#~set(CUML_LIBRARY_DIR /home/thehn/Code/cuml/cpp/build)
#~set(CUML_INCLUDE_DIR /home/thehn/Code/cuml/cpp/include)
#~
#~if(NOT IS_DIRECTORY ${CUML_INCLUDE_DIR})
#~    message(FATAL_ERROR "CUML_INCLUDE_DIR not specified.")
#~endif()
#~if(NOT IS_DIRECTORY ${CUML_LIBRARY_DIR})
#~    message(FATAL_ERROR "CUML_LIBRARY_DIR not specified.")
#~endif()

### --- DBG
### print all available cmake variables
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

#add_definitions(-std=c++11)
#link_directories(/usr/local/lib)
### --- DBG

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/InstanceStixels/src/)
set(SOURCES
    ${SOURCE_DIR}/RoadEstimation.cu
    ${SOURCE_DIR}/RoadEstimationKernels.cu 
    ${SOURCE_DIR}/Stixels.cu
    ${SOURCE_DIR}/StixelsKernels.cu
    ${SOURCE_DIR}/TRTOnnxCNN.cpp)

# Set cuda flags
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 --use_fast_math
    -maxrregcount=72 # 80 registers results in request for too many resources
    #-std=c++11
    #-gencode=arch=compute_30,code=sm_30
    #-gencode=arch=compute_35,code=sm_35
    #-gencode=arch=compute_50,code=sm_50
    #-gencode=arch=compute_52,code=sm_52
    -gencode=arch=compute_61,code=sm_61
    -gencode=arch=compute_70,code=sm_70
    -gencode=arch=compute_75,code=sm_75
    #-lineinfo
    #--resource-usage
    #-g -G
    )

set(InstanceStixels_INCLUDE_DIRS
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    InstanceStixels/include/InstanceStixels)
set(InstanceStixels_LIBS
    ${TENSORRT_LIBRARY}
    cuml++)

cuda_add_library(InstanceStixels STATIC
    ${SOURCES})
target_include_directories(InstanceStixels PRIVATE
    ${InstanceStixels_INCLUDE_DIRS})
#target_link_directories(InstanceStixels PRIVATE
#    ${InstanceStixels_INCLUDE_DIRS})
target_link_libraries(InstanceStixels
    ${InstanceStixels_LIBS})

# PRIVATE keyword omitted because FindCUDA doesn't use it -> error

if(${HDF5_FOUND})
  message("HDF5 found")
  message("Configuring run_cityscapes executable")
  # cityscapes stixel executable
  cuda_add_executable(run_cityscapes
    apps/run_cityscapes.cu
    ${SOURCE_DIR}/H5Segmentation.cpp)

  ## Need to set linker language to CUDA to link the CUDA Runtime
  #set_target_properties(run_cityscapes PROPERTIES LINKER_LANGUAGE "CUDA")

  target_include_directories(run_cityscapes PRIVATE
    ${InstanceStixels_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIR}
    apps/) # rapidjson
  #target_link_directories(run_cityscapes PRIVATE 
  #  ${InstanceStixels_LIB_DIRS})
  # PRIVATE keyword omitted because FindCUDA doesn't use it -> error
  target_link_libraries(run_cityscapes
    ${InstanceStixels_LIBS}
    ${OpenCV_LIBS}
    ${HDF5_LIBRARIES}
    InstanceStixels)
else()
  message("HDF5 not found")
  message("Not configuring run_cityscapes executable")
endif()
if(${catkin_FOUND})
  message("Catkin found")
  message("Configuring ros node")
  project(instance_stixels)

  generate_dynamic_reconfigure_options(
    cfg/InstanceStixels.cfg
    cfg/InstanceStixelsVisualization.cfg
    )

  catkin_package(
    INCLUDE_DIRS InstanceStixels/include
    LIBRARIES InstanceStixels
    CATKIN_DEPENDS
      roscpp
      sensor_msgs
      std_msgs
      instance_stixels_msgs
      visualization_msgs
      message_generation
      cv_bridge
      image_transport
      image_geometry
    #  DEPENDS ${InstanceStixels_LIBS}
    )

  # --- Node for stixel computation
  cuda_add_executable(instance_stixels
    apps/stixels_node_main.cu
    apps/stixels_node.cu
    apps/stixels_wrapper.cu)

  ## Need to set linker language to CUDA to link the CUDA Runtime
  #set_target_properties(instance_stixels PROPERTIES LINKER_LANGUAGE "CUDA")

  add_dependencies(instance_stixels ${${PROJECT_NAME}_EXPORTED_TARGETS})
  target_include_directories(instance_stixels PRIVATE
    ${InstanceStixels_INCLUDE_DIRS}
    InstanceStixels/include/InstanceStixels
    ${catkin_INCLUDE_DIRS})
  #target_link_directories(instance_stixels PRIVATE
  #  ${InstanceStixels_LIB_DIRS})
  # PRIVATE keyword omitted because FindCUDA doesn't use it -> error
  target_link_libraries(instance_stixels
    ${InstanceStixels_LIBS}
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    InstanceStixels)

  # --- Node for stixel visualization
  message("Catkin include dirs: ${catkin_INCLUDE_DIRS}")
  add_executable(ros_stixel_visualization
    apps/visualization_node_main.cpp
    apps/visualization_node.cpp)

  add_dependencies(ros_stixel_visualization ${${PROJECT_NAME}_EXPORTED_TARGETS})
  target_include_directories(ros_stixel_visualization PRIVATE
    ${catkin_INCLUDE_DIRS})
  target_link_libraries(ros_stixel_visualization
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES})
else()
  message("Catkin not found")
  message("Not configuring ros node")
endif()

# Tests are not valid anymore since CNN output is used as int32.
#~ # test executable
#~ set(CUDA_NVCC_FLAGS
#~     ${CUDA_NVCC_FLAGS}
#~     #-g -G
#~     )
#~ include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#~ include_directories(${HDF5_INCLUDE_DIR})
#~ cuda_add_executable(run_tests 
#~     ${TEST_SOURCES} ${TESTED_SOURCES})
#~ target_link_libraries(run_tests ${OpenCV_LIBS} ${HDF5_LIBRARIES})
#~ 
#~ enable_testing()
#~ add_test(NAME Tests COMMAND run_tests)
