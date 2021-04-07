// This file is part of Instance Stixels:
// https://github.com/tudelft-iv/instance-stixels
//
// Copyright (c) 2019 Thomas Hehn.
//
// Instance Stixels is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Instance Stixels is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Instance Stixels. If not, see <http://www.gnu.org/licenses/>.

#include <algorithm>

#include "catch.hpp"
#include "testdata.h"

#include "util.h"
#include "StixelsKernels.h"
#include "Segmentation.h"

#define PRINT_VAR(var) std::cout << #var << " = " << var << "\n";

// TODO: restrict d_segmentation
__global__ void ComputePrefixSumTestKernel(
        const int classes, const int rows_power2, 
        float* __restrict__ d_segmentation){
	const int col = blockIdx.x;

    for(int c = 0; c < classes; c++){
        ComputePrefixSum(&d_segmentation[col*rows_power2*classes+c*rows_power2],
                         rows_power2);
    }
}

bool test_joincolumnssegmentation(const int rows, const int cols, 
        const int classes, const int realcols, const int rows_power2,
        const int stixel_width, const int width_margin, 
        float* d_segmentation_big, float* d_segmentation){

    // Debug output
    std::cout << "blocks = " << divUp(rows*realcols, 256)
              << ", threads = " << 256 << "\n";

    // Launch test kernel
    JoinColumnsSegmentation<<<divUp(rows*realcols, 256), 256>>>(
    		d_segmentation_big, d_segmentation, stixel_width,
    		width_margin, rows, cols, classes, realcols, rows_power2);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // Copy result to host
    std::vector<float> segmentation_joined(rows_power2*realcols*classes);
	CUDA_CHECK_RETURN(cudaMemcpy(segmentation_joined.data(), d_segmentation,
                                 sizeof(float)*rows_power2*realcols*classes,
                                 cudaMemcpyDeviceToHost));

    // Debug output
    for(int i = 0; i < TESTDATA::joined.size(); i++){
        std::cout << segmentation_joined[i] << " - " << TESTDATA::joined[i] 
                  << " = " << (segmentation_joined[i] - TESTDATA::joined[i])
                  << " = " << (segmentation_joined[i] == TESTDATA::joined[i])
                  << "\n";
    }
    std::cout << "\n";

    // Comparing difference against 1e-6 is fine, since values are in
    // similar range.
    // https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    return std::equal(TESTDATA::joined.begin(), TESTDATA::joined.end(), 
                      segmentation_joined.begin(),
                      [] (const float& a, const float& b) 
                        { return std::abs(a - b) < 1e-6; } );
}

bool test_computeprefixsum(const int rows, const int cols, const int classes, 
        const int realcols, const int rows_power2, float* d_segmentation){
    // Launch test kernel
    ComputePrefixSumTestKernel<<<realcols, rows>>>(
            classes, rows_power2, d_segmentation);

    // Copy result to host
    std::vector<float> segmentation_prefixsums(rows_power2*realcols*classes);
	CUDA_CHECK_RETURN(cudaMemcpy(segmentation_prefixsums.data(), d_segmentation,
                                 sizeof(float)*rows_power2*realcols*classes,
                                 cudaMemcpyDeviceToHost));

    // Debug output
    for(int i = 0; i < TESTDATA::prefixsums.size(); i++){
        std::cout << segmentation_prefixsums[i] << " - " << TESTDATA::prefixsums[i] 
                  << " = " << (segmentation_prefixsums[i] - TESTDATA::prefixsums[i])
                  << " = " << (segmentation_prefixsums[i] == TESTDATA::prefixsums[i])
                  << "\n";
    }
    std::cout << "\n";

    return std::equal(TESTDATA::prefixsums.begin(), TESTDATA::prefixsums.end(),
                      segmentation_prefixsums.begin(),
                      [] (const float& a, const float& b) 
                        { return std::abs(a - b) < 1e-6; } );
}

TEST_CASE("Testing stixelskernels.", "[stixelskernels]"){
    // Setup test data.
    Segmentation segmentation(TESTDATA::h5_filename.c_str());

    const std::vector<hsize_t> shape = segmentation.get_shape();
    const int stixel_width = TESTDATA::stixel_width;
    const int rows = static_cast<int>(shape[0]);
    const int cols = static_cast<int>(shape[1]);
    const int classes = static_cast<int>(shape[2]);
    const int width_margin = 0;
    const int realcols = (cols-width_margin) / stixel_width;
    const int rows_power2 = (int) powf(2, ceilf(log2f(rows+1)));
    PRINT_VAR(rows);
    PRINT_VAR(cols);
    PRINT_VAR(realcols);
    PRINT_VAR(classes);

    // Move data to GPU
    float *d_segmentation_big;
    float *d_segmentation;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_segmentation_big, 
                                 rows*cols*classes*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_segmentation, 
                                 rows_power2*realcols*classes*sizeof(float)));

    CUDA_CHECK_RETURN(cudaMemcpy(d_segmentation_big, segmentation.data(),
                             sizeof(float)*rows*cols*classes,
                             cudaMemcpyHostToDevice));

    // Do the actual tests
    REQUIRE(test_joincolumnssegmentation(rows, cols, classes, realcols,
                                         rows_power2, stixel_width,
                                         width_margin, d_segmentation_big,
                                         d_segmentation)
            == true);

    REQUIRE(test_computeprefixsum(rows, cols, classes, realcols, rows_power2,
                                  d_segmentation)
            == true);

    // Free all the memory!
    CUDA_CHECK_RETURN(cudaFree(d_segmentation_big));
    CUDA_CHECK_RETURN(cudaFree(d_segmentation));
}
