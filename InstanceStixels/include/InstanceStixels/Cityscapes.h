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

// TODO: make this file more generic, and less specific to cityscapes.
// -> put cityscapes specifics in separate header file.
#ifndef CITYSCAPES_H_
#define CITYSCAPES_H_

#include "configuration.h" // pixel_t, MAX_LOGPROB

// Compute the sum based on the downsampled prefix sum.
// TODO: __restrict__ pointers?
__inline__ __device__ int DownsampledSum(const int32_t* column_start,
        const int vB, const int vT){
    // Note: we need to correct for remainders due to downsizing.
    // Assume the entire stixel shifted to a multiple of the DOWNSAMPLE_FACTOR
    // and then correct for the remainders in the lowest and highest index.
    const int vTmod = vT % DOWNSAMPLE_FACTOR;
    const int vTdiv = vT / DOWNSAMPLE_FACTOR;
    const int vBmod = vB % DOWNSAMPLE_FACTOR;
    const int vBdiv = vB / DOWNSAMPLE_FACTOR;

    return
       (column_start[vTdiv] - column_start[vBdiv]) * DOWNSAMPLE_FACTOR
       + (column_start[vTdiv+1] - column_start[vTdiv]) * (vTmod+1)
       - (column_start[vBdiv+1] - column_start[vBdiv]) * vBmod;
}

__inline__ __device__ float GetGroundSegmentationCost(
        const int32_t* column_start, const int vB, const int vT,
        const int rows_power2){
    const float cost_road = DownsampledSum(&column_start[0], vB, vT);
    // Sidewalk is class id 1, thus shift by rows_power2.
    const float cost_sidewalk = DownsampledSum(&column_start[rows_power2], vB, vT);
    return fminf(cost_road, cost_sidewalk);
}
__inline__ __device__ int GetGroundSegmentationClass(
        const int32_t* column_start, const int vB, const int vT,
        const int rows_power2){
    const float cost_road = DownsampledSum(&column_start[0], vB, vT);
    const float cost_sidewalk = DownsampledSum(&column_start[rows_power2], vB, vT);
    return (cost_road < cost_sidewalk) ? 0 : 1;
}

// Object segmentation functions
__inline__ __device__ float GetObjectSegmentationCost(
        const int32_t* column_start, const int vB, const int vT,
        const int rows_power2,
        const float instance_cost = 0,
        const float non_instance_cost = 0){
    float min_cost_segmentation = MAX_LOGPROB;
    for(int c = 2; c < 19; c++){
        float cost_segmentation = 0.0f;
        if(c < 10){ // non instance classes
            cost_segmentation += non_instance_cost;
        }
        else if(c == 10){ // skip sky
            continue;
        }
        else {// (c > 10) // instance classes
            cost_segmentation += instance_cost;
        }
        cost_segmentation +=
            DownsampledSum(&column_start[c*rows_power2], vB, vT);
        if(min_cost_segmentation > cost_segmentation)
            min_cost_segmentation = cost_segmentation;
    }
    return min_cost_segmentation;
}
__inline__ __device__ int GetObjectSegmentationClass(
        const int32_t* column_start, const int vB, const int vT,
        const int rows_power2,
        const float instance_cost = 0,
        const float non_instance_cost = 0){
    float min_cost_segmentation = MAX_LOGPROB;
    int min_class = 2;
    for(int c = 2; c < 19; c++){
        float cost_segmentation = 0.0f;
        if(c < 10){ // non instance classes
            cost_segmentation += non_instance_cost;
        }
        else if(c == 10){ // skip sky
            continue;
        }
        else {// (c > 10) // instance classes
            cost_segmentation += instance_cost;
        }
        cost_segmentation +=
            DownsampledSum(&column_start[c*rows_power2], vB, vT);
        if(min_cost_segmentation > cost_segmentation){
            min_cost_segmentation = cost_segmentation;
            min_class = c;
        }
    }
    return min_class;
}

// Sky segmentation functions
__inline__ __device__ float GetSkySegmentationCost(
        const int32_t* column_start, const int vB, const int vT,
        const int rows_power2){
    return DownsampledSum(&column_start[10*rows_power2], vB, vT);
}
__inline__ __device__ int GetSkySegmentationClass(
        const int32_t* column_start, const int vB, const int vT,
        const int rows_power2){
    return 10;
}

#endif
