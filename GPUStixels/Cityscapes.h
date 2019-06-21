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

#ifndef CITYSCAPES_H_
#define CITYSCAPES_H_

#include "configuration.h" // pixel_t, MAX_LOGPROB

// TODO: __restrict__?
__inline__ __device__ float GetGroundSegmentationCost(
        const pixel_t* column_start, const int vB, const int vT,
        const int rows_power2){
    const float cost_road =
        column_start[vT+1] - column_start[vB];
    const float cost_sidewalk =
        column_start[1*rows_power2+vT+1] - column_start[1*rows_power2+vB];
    return fminf(cost_road, cost_sidewalk);
}
__inline__ __device__ int GetGroundSegmentationClass(
        const pixel_t* column_start, const int vB, const int vT,
        const int rows_power2){
    const float cost_road =
        column_start[vT+1] - column_start[vB];
    const float cost_sidewalk =
        column_start[1*rows_power2+vT+1] - column_start[1*rows_power2+vB];
    return (cost_road < cost_sidewalk) ? 0 : 1;
}

// Object segmentation functions
__inline__ __device__ float GetObjectSegmentationCost(
        const pixel_t* column_start, const int vB, const int vT,
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
            column_start[c*rows_power2+vT+1] - column_start[c*rows_power2+vB];
        if(min_cost_segmentation > cost_segmentation)
            min_cost_segmentation = cost_segmentation;
    }
    return min_cost_segmentation;
}
__inline__ __device__ int GetObjectSegmentationClass(
        const pixel_t* column_start, const int vB, const int vT,
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
                column_start[c*rows_power2+vT+1] - column_start[c*rows_power2+vB];
        if(min_cost_segmentation > cost_segmentation){
            min_cost_segmentation = cost_segmentation;
            min_class = c;
        }
    }
    return min_class;
}

// Sky segmentation functions
__inline__ __device__ float GetSkySegmentationCost(
        const pixel_t* column_start, const int vB, const int vT,
        const int rows_power2){
    return column_start[10*rows_power2+vT+1] - column_start[10*rows_power2+vB];
}
__inline__ __device__ int GetSkySegmentationClass(
        const pixel_t* column_start, const int vB, const int vT,
        const int rows_power2){
    return 10;
}

#endif
