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

cmake_minimum_required(VERSION 2.4)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Set variables
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(TEST_SOURCES tests/main.cpp 
    tests/segmentation_tests.cpp tests/stixelskernels_tests.cu)
set(TESTED_SOURCES Segmentation.cpp StixelsKernels.cu)

# Find libraries
project(stixels)
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )
find_package( HDF5 REQUIRED CXX )

# generate test data if nonexistent
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/testdata.h")
    message("${CMAKE_CURRENT_SOURCE_DIR}/tests/testdata.h not found. "
            "Executing python script.")
    execute_process(COMMAND python3 generate_testdata.py WORKING_DIRECTORY
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/)
endif()

include_directories(${HDF5_INCLUDE_DIR})

# Set cuda flags
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 --use_fast_math
    #-gencode=arch=compute_30,code=sm_30
    #-gencode=arch=compute_35,code=sm_35
    #-gencode=arch=compute_50,code=sm_50
    #-gencode=arch=compute_52,code=sm_52
    #-gencode=arch=compute_61,code=sm_61
    -gencode=arch=compute_70,code=sm_70
    -lineinfo
    #--resource-usage
    #-g -G
    )

# main stixel executable
cuda_add_executable(
    stixels
    main.cu 
    RoadEstimation.cu RoadEstimationKernels.cu 
    Stixels.cu StixelsKernels.cu Segmentation.cpp)
target_link_libraries(stixels ${OpenCV_LIBS} ${HDF5_LIBRARIES})

# test executable
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -g -G
    )
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HDF5_INCLUDE_DIR})
cuda_add_executable(run_tests 
    ${TEST_SOURCES} ${TESTED_SOURCES})
target_link_libraries(run_tests ${OpenCV_LIBS} ${HDF5_LIBRARIES})

enable_testing()
add_test(NAME Tests COMMAND run_tests)
