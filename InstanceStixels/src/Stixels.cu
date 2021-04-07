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

#include "Stixels.hpp"
//#include <chrono>
#include <fstream>
#include <iostream>

#include <nvToolsExt.h>
#include <cuml/cluster/dbscan.hpp>
#include <cuml/cuml.hpp>


Stixels::Stixels() {
}

Stixels::~Stixels() {
}


/**
  Initialization after setting all the Parameters. Also allocating GPU memory.
**/
void Stixels::Initialize() {
    m_realcols = (m_cols-m_width_margin)/m_column_step;

    m_max_sections = MAX_STIXELS_PER_COLUMN;
    m_instance_classes = 8; // TOOD: Remove dataset specific constants

    //m_stixels = new Section[m_realcols*m_max_sections];
    m_instance_labels = new int[m_instance_classes*m_realcols*m_max_sections];
    m_instance_indices = new int[m_instance_classes*m_realcols*m_max_sections*2];
    m_instances_per_class = new int[m_instance_classes];
    CUDA_CHECK_RETURN(
            cudaMalloc((void**) &d_stixels,
                       m_realcols * m_max_sections * sizeof(Section)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void**) &d_instance_centerofmass,
                       m_instance_classes * m_realcols * m_max_sections * 2
                       * sizeof(float)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void**) &d_instance_labels,
                       m_instance_classes * m_realcols * m_max_sections
                       * sizeof(int32_t)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void**) &d_instance_indices,
                       m_instance_classes * m_realcols * m_max_sections * 2
                       * sizeof(int32_t)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void**) &d_instance_core_candidates,
                       m_instance_classes * m_realcols * m_max_sections
                       * sizeof(bool)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void**) &d_instances_per_class,
                       m_instance_classes/*==TODO: classes*/ * sizeof(int32_t)));

    m_ground_function = new float[m_rows];

    // Log LUT, range: 0.0f - 1.0f
    m_log_lut = new float[LOG_LUT_SIZE+1];
    for(int i = 0; i < LOG_LUT_SIZE; i++) {
        const float log_res = (float)i/((float)LOG_LUT_SIZE);
        m_log_lut[i] = logf (log_res);
    }
    m_log_lut[LOG_LUT_SIZE] = 0.0f;
    // NOTE:
    // m_log_lut[0] = -inf -> m_normalization_ground["high idx"] = -inf
    // Should not be relevant though, as high indices (i.e. upper rows) are
    // above horizon and not considered ground anymore.
    // Thus, GetDataCostGround is not called with any of these "-inf"
    // normalizaton_ground values.

    // Frequently used values
    m_max_dis_log = logf(m_max_disf);
    m_rows_log = logf((float)m_rows);
    m_puniform_sky = m_max_dis_log - logf(m_pout_sky);
    m_puniform = m_max_dis_log - logf(m_pout);
    m_pnexists_given_sky_log = -logf(m_pnexists_given_sky);
    m_nopnexists_given_sky_log = -logf(1.0f-m_pnexists_given_sky);
    m_pnexists_given_ground_log = -logf(m_pnexists_given_ground);
    m_nopnexists_given_ground_log = -logf(1.0f-m_pnexists_given_ground);
    m_pnexists_given_object_log = -logf(m_pnexists_given_object);
    m_nopnexists_given_object_log = -logf(1.0f-m_pnexists_given_object);

    // Data term precomputation
    m_normalization_ground = new float[m_rows];
    m_inv_sigma2_ground = new float[m_rows];
    m_normalization_object = new float[m_max_dis];
    m_inv_sigma2_object = new float[m_max_dis];
    m_object_disparity_range = new float[m_max_dis];

    for(int i = 0; i < m_max_dis; i++) {
        float previous_mean = (float) i;
        m_object_disparity_range[i] =
            ComputeObjectDisparityRange(previous_mean);
    }

    // Precomputation of data term
    PrecomputeSky();
    PrecomputeObject();

    // Object Data Cost LUT
    m_obj_cost_lut = new float[m_max_dis*m_max_dis];

    for(int fn = 0; fn < m_max_dis; fn++) {
        for(int dis = 0; dis < m_max_dis; dis++) {
            m_obj_cost_lut[fn*m_max_dis+dis] =
                GetDataCostObject(fn, dis);
        }
    }

    const int rows_power2 = (int) powf(2, ceilf(log2f(m_rows+1)));
    const int rows_power2_segmentation =
        (int) powf(2, ceilf(log2f(m_rows/8 + 1)));

    // Malloc
    CUDA_CHECK_RETURN(
            cudaMalloc((void **)&d_segmentation,
                       rows_power2_segmentation*m_realcols*m_segmentation_channels
                       *sizeof(int32_t)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **)&d_disparity_big,
                       m_rows*m_cols*sizeof(float)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **)&d_disparity,
                       m_rows*m_realcols*sizeof(float)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **)&d_ground_function,
                       m_rows*m_realcols*sizeof(float)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **)&d_normalization_ground,
                       m_rows*sizeof(float)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **)&d_inv_sigma2_ground,
                       m_rows*sizeof(float)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **)&d_object_disparity_range,
                       m_max_dis*sizeof(float)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **)&d_object_lut,
                       (rows_power2+1)*m_realcols*m_max_dis*sizeof(float)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **)&d_obj_cost_lut,
                       m_max_dis*m_max_dis*sizeof(float)));

    int current_device;
    cudaGetDevice(&current_device);
    int compute_capability_minor;
    cudaDeviceGetAttribute(&compute_capability_minor,
                           cudaDevAttrComputeCapabilityMinor,
                           current_device);
    int compute_capability_major;
    cudaDeviceGetAttribute(&compute_capability_major,
                           cudaDevAttrComputeCapabilityMajor,
                           current_device);

    // Volta is theoretically able to handle 96KB.
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-7-x
    // Originally:
    // 4 * (1024 * 6 + 128) + 2 * (1024 * 3) + 4 * 1024 + 4 * (1024 * 2)
    // = 43520 bytes
    const int instance_workspace_size = sizeof(int64_t)*(rows_power2) * 4;
    m_shared_mem_size =
        sizeof(float)*(rows_power2*6+m_max_dis)
        + sizeof(int16_t)*(rows_power2*3)
        + sizeof(pixel_t)*(rows_power2) // sum
        + sizeof(pixel_t)*(rows_power2) // valid
        + instance_workspace_size; // 4 * instance channels

    d_instance_meansx_ps = NULL; // use shared memory instead
    if(compute_capability_major != 7 || compute_capability_minor != 0){
        m_shared_mem_size -= instance_workspace_size;
        //printf("Allocating device memory.\n");
        // Allocate instance workspace for each thread block!
        CUDA_CHECK_RETURN(
                cudaMalloc((void**) &d_instance_meansx_ps,
                           instance_workspace_size * m_realcols));
    }
    //printf("Shared mem size: %d\n", m_shared_mem_size);

    // Memcpy
    CUDA_CHECK_RETURN(
            cudaMemcpy(d_object_disparity_range,
                       m_object_disparity_range,
                       sizeof(float)*m_max_dis,
                       cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(
            cudaMemcpy(d_obj_cost_lut,
                       m_obj_cost_lut,
                       sizeof(float)*m_max_dis*m_max_dis,
                       cudaMemcpyHostToDevice));

    m_params.rows = m_rows;
    m_params.cols = m_realcols;
    m_params.max_dis = m_max_dis;
    m_params.invalid_disparity = m_invalid_disparity;
    m_params.rows_log = m_rows_log;
    m_params.pnexists_given_sky_log = m_pnexists_given_sky_log;
    m_params.normalization_sky = m_normalization_sky;
    m_params.inv_sigma2_sky = m_inv_sigma2_sky;
    m_params.puniform_sky = m_puniform_sky;
    m_params.nopnexists_given_sky_log = m_nopnexists_given_sky_log;
    m_params.pnexists_given_ground_log = m_pnexists_given_ground_log;
    m_params.puniform = m_puniform;
    m_params.nopnexists_given_ground_log = m_nopnexists_given_ground_log;
    m_params.pnexists_given_object_log = m_pnexists_given_object_log;
    m_params.nopnexists_given_object_log = m_nopnexists_given_object_log;
    m_params.baseline = m_baseline;
    m_params.focal = m_focal;
    m_params.range_objects_z = m_range_objects_z;
    m_params.pord = m_pord;
    m_params.epsilon = m_epsilon;
    m_params.pgrav = m_pgrav;
    m_params.pblg = m_pblg;
    m_params.rows_power2 = rows_power2;
    m_params.rows_power2_segmentation = rows_power2_segmentation;
    m_params.max_sections = m_max_sections;
    m_params.max_dis_log = m_max_dis_log;
    m_params.width_margin = m_width_margin;
    m_params.segmentation_classes = m_segmentation_classes;
    m_params.segmentation_channels = m_segmentation_channels;
    m_params.prior_weight = m_prior_weight;
    m_params.disparity_weight = m_disparity_weight;
    m_params.segmentation_weight = m_segmentation_weight;
    m_params.instance_weight = m_instance_weight;
    m_params.column_step = m_column_step;

    m_is_initialized = true;
}

void Stixels::Finish() {
    //delete[] m_stixels;
    delete[] m_instance_labels;
    delete[] m_instance_indices;
    delete[] m_instances_per_class;

    delete[] m_ground_function;
    delete[] m_normalization_ground;
    delete[] m_inv_sigma2_ground;
    delete[] m_normalization_object;
    delete[] m_inv_sigma2_object;
    delete[] m_object_disparity_range;
    delete[] m_obj_cost_lut;
    delete[] m_log_lut;

    CUDA_CHECK_RETURN(cudaFree(d_segmentation));
    CUDA_CHECK_RETURN(cudaFree(d_disparity_big));
    CUDA_CHECK_RETURN(cudaFree(d_disparity));
    CUDA_CHECK_RETURN(cudaFree(d_ground_function));
    CUDA_CHECK_RETURN(cudaFree(d_normalization_ground));
    CUDA_CHECK_RETURN(cudaFree(d_inv_sigma2_ground));
    CUDA_CHECK_RETURN(cudaFree(d_object_disparity_range));
    CUDA_CHECK_RETURN(cudaFree(d_object_lut));
    CUDA_CHECK_RETURN(cudaFree(d_stixels));
    CUDA_CHECK_RETURN(cudaFree(d_instance_centerofmass));
    CUDA_CHECK_RETURN(cudaFree(d_instance_labels));
    CUDA_CHECK_RETURN(cudaFree(d_instance_indices));
    CUDA_CHECK_RETURN(cudaFree(d_instance_core_candidates));
    CUDA_CHECK_RETURN(cudaFree(d_instances_per_class));
    CUDA_CHECK_RETURN(cudaFree(d_obj_cost_lut));
    CUDA_CHECK_RETURN(cudaFree(d_instance_meansx_ps));

    m_is_initialized = false;
}



//////////////////////////////////////////////////////////////////////
// Functions
//////////////////////////////////////////////////////////////////////

// Load a stixel configuration and also check non-default values.
void Stixels::SetConfig(const StixelConfig& config){
    // Check whether necessary values have been set.
    if(config.rows == -1 || config.cols == -1){
        throw std::invalid_argument("Number of rows or columns are not set.");
    }
    if(config.max_dis == -1){
        throw std::invalid_argument("Maximum disparity value is not set.");
    }
    if(config.eps == -1 || config.min_pts == -1 || config.size_filter == -1){
        throw std::invalid_argument("Clustering parameters are not set.");
    }
    if(config.prior_weight == -1 || config.segmentation_weight == -1
       || config.instance_weight == -1 || config.disparity_weight == -1){
        throw std::invalid_argument("Energy term weights are not set.");
    }
    if(config.column_step == -1){
        throw std::invalid_argument("Stixel width is not set.");
    }
    if(config.focal == -1
       || config.baseline == -1){
        throw std::invalid_argument("Camera parameters are not set.");
    }

    SetDisparityParameters(
        config.rows, config.cols, config.max_dis, config.invalid_disparity,
        config.sigma_disparity_object, config.sigma_disparity_ground,
        config.sigma_sky);
    SetSegmentationParameters(
        config.n_semantic_classes, config.n_offset_channels);
    SetClusteringParameters(
        config.eps, config.min_pts, config.size_filter);
    SetWeightParameters(
        config.prior_weight, config.disparity_weight,
        config.segmentation_weight, config.instance_weight);
    SetProbabilities(
        config.pout, config.pout_sky, config.pground_given_nexist,
        config.pobject_given_nexist, config.psky_given_nexist,
        config.pnexist_dis, config.pground, config.pobject, config.psky,
        config.pord, config.pgrav, config.pblg);
    SetModelParameters(
        config.column_step, config.median_join, config.epsilon,
        config.range_objects_z, config.width_margin);
    SetCameraParameters(
        config.focal, config.baseline, config.sigma_camera_tilt,
        config.sigma_camera_height, config.camera_center_x,
        config.camera_center_y);
}

void Stixels::SetSegmentation(const std::vector<int32_t>& segmentation) {
    CUDA_CHECK_RETURN(
            cudaMemcpy(d_segmentation,
                       segmentation.data(),
                       sizeof(int32_t) * segmentation.size(),
                       cudaMemcpyHostToDevice));
}

void Stixels::SetDisparityImage(const std::vector<pixel_t>& disp_im) {
    // New image joining the columns
    CUDA_CHECK_RETURN(
            cudaMemcpyAsync(d_disparity_big,
                            disp_im.data(),
                            sizeof(pixel_t)*disp_im.size(),
                            cudaMemcpyHostToDevice));
}

pixel_t* Stixels::GetInputDisparityImageOnDevice(){
    return d_disparity_big;
}

void Stixels::SetProbabilities(float pout, float pout_sky,
        float pground_given_nexist, float pobject_given_nexist,
        float psky_given_nexist, float pnexist_dis, float pground,
        float pobject, float psky, float pord, float pgrav, float pblg) {
    m_pout = pout;
    m_pout_sky = pout_sky;
    m_pnexists_given_ground = (pground_given_nexist*pnexist_dis)/pground;
    m_pnexists_given_object = (pobject_given_nexist*pnexist_dis)/pobject;
    m_pnexists_given_sky = (psky_given_nexist*pnexist_dis)/psky;
    m_pord = pord;
    m_pgrav = pgrav;
    m_pblg = pblg;
}

void Stixels::SetRoadParameters(int vhor, float camera_tilt,
        float camera_height, float alpha_ground){
    m_vhor = m_rows-vhor-1;
    m_camera_tilt = camera_tilt;
    m_camera_height = camera_height;
    m_alpha_ground = alpha_ground;
}

void Stixels::SetCameraParameters(float focal, float baseline,
        float sigma_camera_tilt, float sigma_camera_height,
        float camera_center_x /*=-1*/, float camera_center_y /*=-1*/) {
    m_focal = focal;
    m_baseline = baseline;
    // Degrees to radians
    m_sigma_camera_tilt = sigma_camera_tilt*(PIFLOAT)/180.0f;
    m_sigma_camera_height = sigma_camera_height;
    m_camera_center_x = camera_center_x;
    m_camera_center_y = camera_center_y;
}

void Stixels::SetClusteringParameters(
        const float eps, const int min_pts, const int size_filter){
    m_params.clustering_eps = eps;
    m_params.clustering_min_pts = min_pts;
    m_params.clustering_size_filter = size_filter;
}

void Stixels::SetSegmentationParameters(
        const int classes, const int instance_channels){
    m_segmentation_classes = classes;
    m_segmentation_channels = classes + instance_channels;
}

void Stixels::SetWeightParameters(const float prior_weight,
        const float disparity_weight, const float segmentation_weight,
        const float instance_weight){
    m_prior_weight = prior_weight;
    m_disparity_weight = disparity_weight;
    m_segmentation_weight = segmentation_weight;

    // Divide out segmentation weight.
    m_instance_weight = 0.0;
    if(segmentation_weight > 1e-5){
        m_instance_weight = instance_weight / segmentation_weight;
        if(instance_weight < 1e-8){
            m_instance_weight = 0.0;
        }
    }
}

void Stixels::SetDisparityParameters(const int rows, const int cols,
        const int max_dis, const float invalid_disparity,
        const float sigma_disparity_object, const float sigma_disparity_ground,
        const float sigma_sky) {
    m_rows = rows;
    m_cols = cols;
    m_max_dis = max_dis;
    m_max_disf = (float) m_max_dis;
    m_sigma_disparity_object = sigma_disparity_object;
    m_sigma_disparity_ground = sigma_disparity_ground;
    m_sigma_sky = sigma_sky;
    m_invalid_disparity = invalid_disparity;
}

void Stixels::SetModelParameters(const int column_step, const bool median_join,
        float epsilon, float range_objects_z, int width_margin) {
    m_column_step = column_step;
    m_median_join = median_join;
    m_epsilon = epsilon;
    m_range_objects_z = range_objects_z;
    m_width_margin = width_margin;
}


float Stixels::Compute(const bool pairwise, StixelsData& stixels_data,
        int32_t* d_segmentation_local) {
    //~ float elapsed_time_ms = 0;
    //~ float elapsed_time_ms_task = 0;
    //~ cudaEvent_t start, stop;
    //~ cudaEventCreate(&start);
    //~ cudaEventCreate(&stop);
    //~ cudaEventRecord(start, 0);
    //
    if(d_segmentation_local == nullptr){
        d_segmentation_local = d_segmentation;
    }

    // Precomputation of data term
    PrecomputeGround();

    // start/stop/print timer
    //~ cudaEventRecord(stop, 0);
    //~ cudaEventSynchronize(stop);
    //~ cudaEventElapsedTime(&elapsed_time_ms_task, start, stop);
    //~ elapsed_time_ms += elapsed_time_ms_task;
    //~ std::cout << "Elapsed time PrecomputeGround: "
    //~           << elapsed_time_ms_task << "\n";
    //~ cudaEventDestroy(start);
    //~ cudaEventDestroy(stop);
    //~ cudaEventCreate(&start);
    //~ cudaEventCreate(&stop);
    //~ cudaEventRecord(start, 0);

    // Copy precomputed LUTs to GPU.
    CUDA_CHECK_RETURN(
            cudaMemcpy(d_ground_function,
                       m_ground_function,
                       sizeof(float)*m_rows,
                       cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(
            cudaMemcpy(d_normalization_ground,
                       m_normalization_ground,
                       sizeof(float)*m_rows,
                       cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(
            cudaMemcpy(d_inv_sigma2_ground,
                       m_inv_sigma2_ground,
                       sizeof(float)*m_rows,
                       cudaMemcpyHostToDevice));

    // start/stop/print timer
    //~ cudaEventRecord(stop, 0);
    //~ cudaEventSynchronize(stop);
    //~ cudaEventElapsedTime(&elapsed_time_ms_task, start, stop);
    //~ elapsed_time_ms += elapsed_time_ms_task;
    //~ std::cout << "Elapsed time copy to dev: "
    //~           << elapsed_time_ms_task << "\n";
    //~ cudaEventDestroy(start);
    //~ cudaEventDestroy(stop);
    //~ cudaEventCreate(&start);
    //~ cudaEventCreate(&stop);
    //~ cudaEventRecord(start, 0);

    // Join disparity values per stixel
    JoinColumns<<<divUp(m_rows*m_realcols, 256), 256>>>(
            d_disparity_big, d_disparity, m_column_step, m_median_join,
            m_width_margin, m_rows, m_cols, m_realcols, m_invalid_disparity);

    // start/stop/print timer
    //~ cudaEventRecord(stop, 0);
    //~ cudaEventSynchronize(stop);
    //~ cudaEventElapsedTime(&elapsed_time_ms_task, start, stop);
    //~ elapsed_time_ms += elapsed_time_ms_task;
    //~ std::cout << "Elapsed time join columns: "
    //~           << elapsed_time_ms_task << "\n";
    //~ cudaEventDestroy(start);
    //~ cudaEventDestroy(stop);
    //~ cudaEventCreate(&start);
    //~ cudaEventCreate(&stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr
            << "Error JoinColumns: " << cudaGetErrorString(err)
            << " (" << err << ")\n";
    }

    m_params.vhor = m_vhor;

    //~ cudaEventRecord(start, 0);
    ComputeObjectLUT<<<m_realcols, 512>>>(
            d_disparity, d_obj_cost_lut, d_object_lut, m_params,
            (int) powf(2, ceilf(log2f(m_rows))));

    CUDA_CHECK_RETURN(cudaMemset(d_instances_per_class, 0,
                                 m_instance_classes*sizeof(int32_t)));
    // start/stop/print timer
    //~ cudaEventRecord(stop, 0);
    //~ cudaEventSynchronize(stop);
    //~ cudaEventElapsedTime(&elapsed_time_ms_task, start, stop);
    //~ elapsed_time_ms += elapsed_time_ms_task;
    //~ std::cout << "Elapsed time computeobjectlut: "
    //~           << elapsed_time_ms_task << "\n";
    //~ cudaEventDestroy(start);
    //~ cudaEventDestroy(stop);
    //~ cudaEventCreate(&start);
    //~ cudaEventCreate(&stop);
    //~ cudaEventRecord(start, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr
            << "Error ComputeObjectLUT: " << cudaGetErrorString(err)
            << " (" << err << ")\n";
    }

    //printf("Shared mem size: %d\n", m_shared_mem_size);

    if(pairwise){
        cudaFuncSetAttribute(
                StixelsKernel<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                m_shared_mem_size);
        StixelsKernel<true><<<m_realcols, m_rows, m_shared_mem_size>>>(
                        d_disparity, d_segmentation_local,
                        m_params, d_ground_function,
                        d_normalization_ground, d_inv_sigma2_ground,
                        d_object_disparity_range, d_object_lut, d_stixels,
                        d_instance_centerofmass, d_instance_indices,
                        d_instance_core_candidates, d_instances_per_class,
                        d_instance_meansx_ps);
    }
    else{
        cudaFuncSetAttribute(
                StixelsKernel<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                m_shared_mem_size);
        StixelsKernel<false><<<m_realcols, m_rows, m_shared_mem_size>>>(
                        d_disparity, d_segmentation_local,
                        m_params, d_ground_function,
                        d_normalization_ground, d_inv_sigma2_ground,
                        d_object_disparity_range, d_object_lut, d_stixels,
                        d_instance_centerofmass, d_instance_indices,
                        d_instance_core_candidates, d_instances_per_class,
                        d_instance_meansx_ps);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr
            << "Error StixelsKernel: " << cudaGetErrorString(err)
            << " (" << err << ")\n";
    }

    // Synchronize
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // start/stop/print timer
    //~ cudaEventRecord(stop, 0);
    //~ cudaEventSynchronize(stop);
    //~ cudaEventElapsedTime(&elapsed_time_ms_task, start, stop);
    //~ elapsed_time_ms += elapsed_time_ms_task;
    //~ std::cout << "Elapsed time stixelskernel: "
    //~           << elapsed_time_ms_task << "\n";
    //~ cudaEventDestroy(start);
    //~ cudaEventDestroy(stop);

    //~ elapsed_time_ms += ClusterInstances();
    ClusterInstances();

    stixels_data.sections.resize(m_realcols*m_max_sections);
    stixels_data.rows = m_rows;
    stixels_data.cols = m_cols;
    stixels_data.realcols = m_realcols;
    stixels_data.max_sections = m_max_sections;
    stixels_data.max_dis = m_max_dis;
    stixels_data.column_step = m_column_step;
    stixels_data.semantic_classes = m_segmentation_classes;
    stixels_data.alpha_ground = m_alpha_ground;
    // Note that the coordinate axis is flipped in SetRoadParameters,
    // thus this corresponds to the use of SaveStixels in the
    // run_cityscapes executable.
    stixels_data.vhor = m_vhor;

    CUDA_CHECK_RETURN(
            cudaMemcpy(stixels_data.sections.data(),
                       d_stixels,
                       m_realcols*m_max_sections*sizeof(Section),
                       cudaMemcpyDeviceToHost));

    //~ return elapsed_time_ms;
    return -1;
}

float Stixels::ClusterInstances(){
    const float eps = m_params.clustering_eps;
    const int min_pts = m_params.clustering_min_pts;

    CUDA_CHECK_RETURN(
            cudaMemcpy(m_instances_per_class,
                       d_instances_per_class,
                       m_instance_classes*sizeof(int),
                       cudaMemcpyDeviceToHost));
    int nCols = 2;
    int max_bytes_per_batch = 0;  // allow algorithm to set this
    // or: (size_t)13e9, which is used as default in example script

    ML::cumlHandle cumlHandle;
    //~ std::chrono::steady_clock::time_point begin =
    //~         std::chrono::steady_clock::now();

    // TODO: Remove dataset specific constants
    for(int class_id = 0; class_id < m_instance_classes; class_id++){
        if(m_instances_per_class[class_id] > 0){
            // Cityscapes: real class class_id + 11
            ML::dbscanFit(cumlHandle,
                          &d_instance_centerofmass[(class_id*m_realcols*m_max_sections)*2],
                          m_instances_per_class[class_id],
                          nCols, eps, min_pts,
                          &d_instance_labels[class_id*m_realcols*m_max_sections],
                          max_bytes_per_batch, false,
                          &d_instance_core_candidates[class_id*m_realcols*m_max_sections]);
        }
    }
    //~ cudaDeviceSynchronize();
    //~ std::chrono::steady_clock::time_point end =
    //~         std::chrono::steady_clock::now();
    //~ int microseconds =
    //~     std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();

    //~ std::cout << "Inference time: "
    //~           << microseconds
    //~           << "\n";

    //~ return microseconds * 1e-3;
    return -1;
}

std::vector<float> Stixels::Get3DVertices(
        const StixelsData& stixels_data){
    if(m_camera_center_x == -1 || m_camera_center_y == -1){
        throw std::invalid_argument("Camera parameters are not set.");
    }

    std::vector<float> vertices;
    for(size_t i = 0; i < m_realcols; i++) {
        for(size_t j = 0; j < m_max_sections; j++) {
            const Section& section = stixels_data.sections[i*m_max_sections+j];
            if(section.type == -1){
                break;
            }

            const float x_l = i*m_column_step;
            const float x_r = x_l + m_column_step;
            const float y_t = m_rows - section.vT - 1;
            const float y_b = m_rows - section.vB;

            // TODO: try std::numeric_limits<float>::quiet_NaN(); instead.
            float top_depth = 0.0; // If SKY, then points are at 0.0.
            float bottom_depth = 0.0;
            if(section.type == OBJECT) {
                top_depth = m_baseline * m_focal / section.disparity;
                bottom_depth = top_depth;
            }
            else if(section.type == GROUND) {
                const float top_disparity =
                    stixels_data.alpha_ground
                    * (stixels_data.vhor - section.vT);
                const float bottom_disparity =
                    stixels_data.alpha_ground
                    * (stixels_data.vhor - section.vB);

                top_depth = m_baseline * m_focal / top_disparity;
                bottom_depth = m_baseline * m_focal / bottom_disparity;
            }
            // Start at top left, then clockwise
            vertices.push_back(- top_depth/m_focal * (m_camera_center_x - x_l));
            vertices.push_back(- top_depth/m_focal * (m_camera_center_y - y_t));
            vertices.push_back(top_depth);

            // top right
            vertices.push_back(- top_depth/m_focal * (m_camera_center_x - x_r));
            vertices.push_back(- top_depth/m_focal * (m_camera_center_y - y_t));
            vertices.push_back(top_depth);

            // bottom right
            vertices.push_back(- bottom_depth/m_focal * (m_camera_center_x - x_r));
            vertices.push_back(- bottom_depth/m_focal * (m_camera_center_y - y_b));
            vertices.push_back(bottom_depth);

            // bottom left
            vertices.push_back(- bottom_depth/m_focal * (m_camera_center_x - x_l));
            vertices.push_back(- bottom_depth/m_focal * (m_camera_center_y - y_b));
            vertices.push_back(bottom_depth);
        }
    }
    return vertices;
}

std::map<std::pair<int,int>,int> Stixels::GetInstanceStixels() {
    // Note: these copy commands require additional ~0.8 milliseconds.
    // However, they can be done asynchronously, when processing mutliple
    // frames.
    CUDA_CHECK_RETURN(
            cudaMemcpy(m_instance_labels,
                       d_instance_labels,
                       m_instance_classes*m_realcols*m_max_sections
                       * sizeof(int32_t),
                       cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(
            cudaMemcpy(m_instance_indices,
                       d_instance_indices,
                       m_instance_classes*m_realcols*m_max_sections * 2
                       * sizeof(int32_t),
                       cudaMemcpyDeviceToHost));

    std::map<std::pair<int,int>,int> instance_stixels_mapping;
    for(int class_id = 0; class_id < m_instance_classes; class_id++){
        const int instances = m_instances_per_class[class_id];
        if(instances > 0){
            const int class_offset = class_id*m_realcols*m_max_sections;
            for(int i = 0; i < instances; i++){
                const int u = m_instance_indices[class_offset*2 + i*2];
                const int v = m_instance_indices[class_offset*2 + i*2 + 1];
                const int label = m_instance_labels[class_offset+i];
                instance_stixels_mapping[std::make_pair(u,v)] = label;
            }
        }
    }

    return instance_stixels_mapping;
}

int Stixels::GetRealCols() {
    return m_realcols;
}

int Stixels::GetMaxSections() {
    return m_max_sections;
}

float Stixels::FastLog(float v) {
    return m_log_lut[(int) ( (v)*LOG_LUT_SIZE + 0.5f )];
}

void Stixels::PrecomputeGround() {
    const float fb = (m_focal*m_baseline) / m_camera_height;
    const float pout = m_pout;

    for(int v = 0; v < m_rows; v++) {
        const float fn = GroundFunction(v);
        m_ground_function[v] = fn;

        const float x = m_camera_tilt + (float)(m_vhor-v) / m_focal;
        const float sigma2_road =
            fb * fb *
            ( m_sigma_camera_height * m_sigma_camera_height * x * x
              / (m_camera_height * m_camera_height)
              + m_sigma_camera_tilt * m_sigma_camera_tilt );
        const float sigma =
            sqrtf( m_sigma_disparity_ground * m_sigma_disparity_ground
                   + sigma2_road );

        const float a_range =
            0.5f * ( erf( (m_max_disf-fn) / (sigma*sqrtf(2.0f)) )
                     - erf( (-fn) / (sigma*sqrtf(2.0f)) ) );

        m_normalization_ground[v] =
            FastLog(a_range) - FastLog( (1.0f-pout)
                                        / (sigma * sqrtf(2.0f*PIFLOAT)) );
        m_inv_sigma2_ground[v] = 1.0f / (2.0f*sigma*sigma);
    }
}

void Stixels::PrecomputeObject() {
    const float pout = m_pout;

    for(int dis = 0; dis < m_max_dis; dis++) {
        const float fn = (float) dis;

        const float sigma_object =
            fn * fn * m_range_objects_z / (m_focal*m_baseline);
        const float sigma =
            sqrtf( m_sigma_disparity_object * m_sigma_disparity_object
                   + sigma_object * sigma_object );

        const float a_range =
            0.5f * ( erf( (m_max_disf-fn) / (sigma*sqrtf(2.0f)) )
                     - erf( (-fn) / (sigma*sqrtf(2.0f)) ) );

        m_normalization_object[dis] =
            FastLog(a_range) - FastLog( (1.0f - pout)
                                        / (sigma * sqrtf(2.0f*PIFLOAT)) );
        m_inv_sigma2_object[dis] = 1.0f / (2.0f * sigma * sigma);
    }
}

float Stixels::GetDataCostObject(const int fn, const int dis){
    float data_cost = m_pnexists_given_object_log;
    if(dis != (int) m_invalid_disparity) {
        const float model_diff = (float) (dis-fn);
        const float pgaussian =
            m_normalization_object[fn]
            + model_diff * model_diff * m_inv_sigma2_object[fn];

        const float p_data = fminf(m_puniform, pgaussian);
        data_cost = p_data + m_nopnexists_given_object_log;
    }
    return data_cost;
}

void Stixels::PrecomputeSky() {
    const float sigma = m_sigma_sky;
    const float pout = m_pout_sky;

    const float a_range =
        0.5f * ( erf( m_max_disf / (sigma*sqrtf(2.0f)) ) - erf(0.0f) );
    m_normalization_sky =
        FastLog(a_range) - logf( (1.0f - pout) / (sigma*sqrtf(2.0f*PIFLOAT)) );
    m_inv_sigma2_sky = 1.0f / (2.0f*sigma*sigma);
}

float Stixels::GroundFunction(const int v) {
    // expects v as in row in [0,h-1]
    //  ^
    //  |
    //  |
    // row
    //  |
    //  |
    // 0/0--column-->
    return m_alpha_ground*(float)(m_vhor-v);
}

float Stixels::ComputeObjectDisparityRange(const float previous_mean) {
    float range_disp = 0.0f;
    if(previous_mean != 0) {
        const float pmean_plus_z =
            (m_baseline * m_focal / previous_mean) + m_range_objects_z;
        range_disp = previous_mean - (m_baseline * m_focal / pmean_plus_z);
    }
    return range_disp;
}

void Stixels::SaveStixels(
        Section *stixels,
        std::map<std::pair<int,int>,int> instance_stixels_mapping,
        const float alpha_ground, const int vhor, const int real_cols,
        const int max_segments, const char *fname) {
    std::ofstream fp;
    fp.open (fname, std::ofstream::out | std::ofstream::trunc);
    //fp << "Writing this to a file.\n";
    if(fp.is_open()) {
        for(size_t i = 0; i < real_cols; i++) {
            for(size_t j = 0; j < max_segments; j++) {
                const Section& section = stixels[i*max_segments+j];
                if(section.type == -1) {
                    break;
                }
                fp << section.type << "," << section.vB << "," << section.vT
                    << "," << section.disparity
                    << "," << section.semantic_class
                    << "," << section.cost
                    << "," << section.instance_meanx
                    << "," << section.instance_meany;
                const auto& it =
                    instance_stixels_mapping.find(std::make_pair(i,j));
                if (it != instance_stixels_mapping.end()){
                    fp << "," << (*it).second;
                }
                fp << ";";
            }
            // Column finished
            fp << std::endl;
        }
        // Add groundplane information.
        fp << "groundplane" << alpha_ground << "," << vhor << "\n";
        fp.close();
    } else {
        std::cerr << "Counldn't write file: " << fname << std::endl;
    }
}

