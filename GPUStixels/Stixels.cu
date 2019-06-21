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

Stixels::Stixels() {
}

Stixels::~Stixels() {
}


/**
  Initialization after setting all the Parameters. Also allocating GPU memory.
**/
void Stixels::Initialize() {
    m_realcols = (m_cols-m_width_margin)/m_column_step;

    m_max_sections = 1024;

    m_stixels = new Section[m_realcols*m_max_sections];
    CUDA_CHECK_RETURN(
            cudaMalloc((void**) &d_stixels, 
                       m_realcols * m_max_sections * sizeof(Section)));

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

    // Malloc
    CUDA_CHECK_RETURN(
            cudaMalloc((void **)&d_segmentation_big, 
                       m_rows*m_cols*m_segmentation_channels*sizeof(float)));
    CUDA_CHECK_RETURN(
            cudaMalloc((void **)&d_segmentation, 
                       rows_power2*m_realcols*m_segmentation_channels
                       *sizeof(float)));
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
}

void Stixels::Finish() {
    delete[] m_stixels;

    delete[] m_ground_function;
    delete[] m_normalization_ground;
    delete[] m_inv_sigma2_ground;
    delete[] m_normalization_object;
    delete[] m_inv_sigma2_object;
    delete[] m_object_disparity_range;
    delete[] m_obj_cost_lut;
    delete[] m_log_lut;

    CUDA_CHECK_RETURN(cudaFree(d_segmentation_big));
    CUDA_CHECK_RETURN(cudaFree(d_segmentation));
    CUDA_CHECK_RETURN(cudaFree(d_disparity_big));
    CUDA_CHECK_RETURN(cudaFree(d_disparity));
    CUDA_CHECK_RETURN(cudaFree(d_ground_function));
    CUDA_CHECK_RETURN(cudaFree(d_normalization_ground));
    CUDA_CHECK_RETURN(cudaFree(d_inv_sigma2_ground));
    CUDA_CHECK_RETURN(cudaFree(d_object_disparity_range));
    CUDA_CHECK_RETURN(cudaFree(d_object_lut));
    CUDA_CHECK_RETURN(cudaFree(d_stixels));
    CUDA_CHECK_RETURN(cudaFree(d_obj_cost_lut));
}



//////////////////////////////////////////////////////////////////////
// Functions
//////////////////////////////////////////////////////////////////////

void Stixels::SetSegmentation(pixel_t *segmentation) {
    m_segmentation = segmentation;
    
    // New image joining the columns
    CUDA_CHECK_RETURN(
            cudaMemcpy(d_segmentation_big, 
                       m_segmentation, 
                       sizeof(pixel_t)*m_rows*m_cols*m_segmentation_channels,
                       cudaMemcpyHostToDevice));
}

void Stixels::SetDisparityImage(pixel_t *disp_im) {
    m_disp_im = disp_im;

    // New image joining the columns
    CUDA_CHECK_RETURN(
            cudaMemcpyAsync(d_disparity_big, 
                            m_disp_im, 
                            sizeof(pixel_t)*m_rows*m_cols,
                            cudaMemcpyHostToDevice));
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

void Stixels::SetCameraParameters(int vhor, float focal, float baseline,
        float camera_tilt, float sigma_camera_tilt, float camera_height,
        float sigma_camera_height, float alpha_ground) {
    m_vhor = m_rows-vhor-1;
    m_focal = focal;
    m_baseline = baseline;
    m_camera_tilt = camera_tilt;
    // Degrees to radians
    m_sigma_camera_tilt = sigma_camera_tilt*(PIFLOAT)/180.0f;
    m_camera_height = camera_height;
    m_sigma_camera_height = sigma_camera_height;
    m_alpha_ground = alpha_ground;
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
    m_instance_weight = instance_weight;
}

void Stixels::SetDisparityParameters(const int rows, const int cols, 
        const int max_dis, const float sigma_disparity_object, 
        const float sigma_disparity_ground, float sigma_sky) {
    m_rows = rows;
    m_cols = cols;
    m_max_dis = max_dis;
    m_max_disf = (float) m_max_dis;
    m_sigma_disparity_object = sigma_disparity_object;
    m_sigma_disparity_ground = sigma_disparity_ground;
    m_sigma_sky = sigma_sky;
}

void Stixels::SetModelParameters(const int column_step, const bool median_step,
        float epsilon, float range_objects_z, int width_margin) {
    m_column_step = column_step;
    m_median_step = median_step;
    m_epsilon = epsilon;
    m_range_objects_z = range_objects_z;
    m_width_margin = width_margin;
}


float Stixels::Compute() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Precomputation of data term
    PrecomputeGround();

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

    // Join disparity values per stixel
    JoinColumns<<<divUp(m_rows*m_realcols, 256), 256>>>(
            d_disparity_big, d_disparity, m_column_step, m_median_step, 
            m_width_margin, m_rows, m_cols, m_realcols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
    }

    // Join segmentation values per stixel
    JoinColumnsSegmentation<<<divUp(m_rows*m_realcols, 256), 256>>>(
            d_segmentation_big, d_segmentation, m_column_step, m_width_margin, 
            m_rows, m_cols, m_segmentation_channels, m_realcols, 
            m_params.rows_power2);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
    }

    m_params.vhor = m_vhor;

    ComputeObjectLUT<<<m_realcols, 512>>>(
            d_disparity, d_obj_cost_lut, d_object_lut, m_params,
            (int) powf(2, ceilf(log2f(m_rows))));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
    }

    // Volta is theoretically able to handle 96KB.
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-7-x
    // Originally:
    // 4 * (1024 * 6 + 128) + 2 * (1024 * 3) + 4 * 1024 + 4 * (1024 * 2)
    // = 43520 bytes
    int shared_mem_size = 
        sizeof(float)*(m_params.rows_power2*6+m_params.max_dis)
        + sizeof(int16_t)*(m_params.rows_power2*3)
        + sizeof(pixel_t)*(m_params.rows_power2) // sum
        + sizeof(pixel_t)*(m_params.rows_power2) // valid
        + sizeof(pixel_t)*(m_params.rows_power2) * 4; // 4 * instance channels
    cudaFuncSetAttribute(
            StixelsKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_size);
    printf("Shared mem size: %d\n", shared_mem_size);
    StixelsKernel<<<m_realcols, m_rows, shared_mem_size>>>(
                    d_disparity, d_segmentation, m_params, d_ground_function, 
                    d_normalization_ground, d_inv_sigma2_ground,
                    d_object_disparity_range, d_object_lut, d_stixels);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s %d\n", cudaGetErrorString(err), err);
    }

    // Synchronize
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK_RETURN(
            cudaMemcpy(m_stixels, 
                       d_stixels,
                       m_realcols*m_max_sections*sizeof(Section),
                       cudaMemcpyDeviceToHost));
    return elapsed_time_ms;
}

Section* Stixels::GetStixels() {
    return m_stixels;
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
    if(!ALLOW_INVALID_DISPARITIES || dis != (int) INVALID_DISPARITY) {
        const float model_diff = (float) (dis-fn);
        // TODO: should this be [fn] instead of [dis]?
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
