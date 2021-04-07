// This file is part of Instance Stixels:
// https://github.com/tudelft-iv/instance-stixels
//
// Originally, it was part of stixels:
// https://github.com/dhernandez0/stixels
//
// Copyright (c) 2016 Daniel Hernandez Juarez.
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

#ifndef STIXELS_HPP_
#define STIXELS_HPP_

#include <vector>
#include <stdint.h>
//#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <map>

#include "configuration.h"
#include "StixelsKernels.h"
#include "util.h"
#include "types.h"

constexpr float PIFLOAT = 3.1416f;

class Stixels {
public:
    // Constructors and destructors
    Stixels();
    ~Stixels();

    // Initialize and finalizes
    void Initialize();
    void Finish();

    // Methods
    float Compute(
            bool pairwise, StixelsData& stixels,
            int32_t* d_segmentation_local = nullptr);
    float ClusterInstances();
    std::map<std::pair<int,int>,int> GetInstanceStixels();
    int GetRealCols();
    int GetMaxSections();
    void SetConfig(const StixelConfig& config);
    void SetSegmentation(const std::vector<int32_t>& segmentation);
    void SetSegmentationParameters(
            const int classes, const int instance_channels);
    void SetClusteringParameters(
            const float eps, const int min_pts, const int size_filter);
    void SetWeightParameters(
	    const float prior_weight, const float disparity_weight,
            const float segmentation_weight, const float instance_weight);
    void SetDisparityImage(const std::vector<pixel_t>& disp_im);
    pixel_t* GetInputDisparityImageOnDevice();
    void SetProbabilities(
	    float pout, float pout_sky, float pground_given_nexist,
            float pobject_given_nexist, float psky_given_nexist,
            float pnexist_dis, float pground, float pobject, float psky,
            float pord, float pgrav, float pblg);
    void SetRoadParameters(
            int vhor, float camera_tilt, float camera_height,
            float alpha_ground);
    void SetCameraParameters(
            float focal, float baseline, float sigma_camera_tilt,
            float sigma_camera_height, float camera_center_x = -1,
            float camera_center_y = -1);
    void SetDisparityParameters(
            const int rows, const int cols, const int max_dis,
            const float invalid_disparity,
            const float sigma_disparity_object,
            const float sigma_disparity_ground, float sigma_sky);
    void SetModelParameters(
            const int column_step, const bool median_join,
            float epsilon, float range_objects_z, int width_margin);
    std::vector<float> Get3DVertices(const StixelsData& stixels_data);
    static void SaveStixels(
            Section* stixels,
            std::map<std::pair<int,int>,int> instance_stixels,
            const float alpha_ground, const int vhor,
            const int real_cols, const int max_segments,
            const char* fname);
    bool IsInitialized() { return m_is_initialized; }
// ATTRIBUTES
private:
    // GPU
    pixel_t *d_disparity;
    pixel_t *d_disparity_big;
    int32_t *d_segmentation;
    float *d_ground_function;
    float *d_normalization_ground;
    float *d_inv_sigma2_ground;
    float *d_normalization_object;
    float *d_inv_sigma2_object;
    float *d_object_lut;
    float *d_object_disparity_range;
    float *d_obj_cost_lut;
    Section *d_stixels;
    float *d_instance_centerofmass;
    int32_t *d_instances_per_class;
    int32_t *d_instance_indices;
    bool *d_instance_core_candidates;
    int32_t *d_instance_labels;
    int64_t *d_instance_meansx_ps;
    int m_shared_mem_size;

    StixelParameters m_params;
    int m_max_sections;
    bool m_is_initialized = false;

    // Probabilities
    float m_pout;
    float m_pout_sky;
    float m_pnexists_given_ground;
    float m_pnexists_given_object;
    float m_pnexists_given_sky;
    float m_pord;
    float m_pgrav;
    float m_pblg;

    // Camera parameters
    float m_focal;
    float m_baseline;
    float m_camera_tilt;
    float m_sigma_camera_tilt;
    float m_camera_height;
    float m_sigma_camera_height;
    float m_camera_center_x;
    float m_camera_center_y;
    int m_vhor;

    // Segmentation Parameters
    int m_segmentation_classes;
    int m_segmentation_channels;

    // Weighting Parameters
    float m_prior_weight;
    float m_disparity_weight;
    float m_segmentation_weight;
    float m_instance_weight;

    // Disparity Parameters
    int m_max_dis;
    float m_max_disf;
    float m_invalid_disparity;
    int m_rows, m_cols, m_realcols;
    float m_sigma_disparity_object;
    float m_sigma_disparity_ground;
    float m_sigma_sky;

    // Other model parameters
    int m_column_step;
    bool m_median_join;
    float m_alpha_ground;
    float m_range_objects_z;
    float m_epsilon;
    int m_width_margin;

    // LUTs
    float *m_log_lut;
    float *m_obj_cost_lut;

    // Values of ground function
    float *m_ground_function;

    // Frequently used values
    float m_max_dis_log;
    float m_rows_log;
    float m_pnexists_given_sky_log;
    float m_nopnexists_given_sky_log;
    float m_pnexists_given_ground_log;
    float m_nopnexists_given_ground_log;
    float m_pnexists_given_object_log;
    float m_nopnexists_given_object_log;

    // Data Term precomputation
    float m_puniform;
    float m_puniform_sky;
    float m_normalization_sky;
    float m_inv_sigma2_sky;
    float *m_normalization_ground;
    float *m_inv_sigma2_ground;
    float *m_normalization_object;
    float *m_inv_sigma2_object;
    float *m_object_disparity_range;

    // Result
    //Section *m_stixels;

    // Instance clustering
    int32_t *m_instance_labels;
    int32_t *m_instance_indices;
    int *m_instances_per_class;
    int m_instance_classes;

    // Methods
    void PrecomputeSky();
    void PrecomputeGround();
    void PrecomputeObject();
    float GetDataCostObject(const int fn, const int dis);
    float ComputeObjectDisparityRange(const float previous_mean);
    pixel_t ComputeMean(const int vB, const int vT, const int u);
    float GroundFunction(const int v);
    float FastLog(float v);

};

#endif /* STIXELS_HPP_ */
