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

#include "stixels_wrapper.h"
#include <iostream>
#include <cassert>

StixelsWrapper::StixelsWrapper(std::string onnx_filename){
    // --- Setup stixel configuration.
    stixel_config_.column_step = 8;
    stixel_config_.max_dis = 128;
    stixel_config_.invalid_disparity = -1.0f;
    stixel_config_.n_semantic_classes = 19; // cityscapes
    stixel_config_.n_offset_channels = 2;
    stixel_config_.prior_weight = 1e4f; // unary regularization

    if(onnx_filename.find("DRNDSDoubleSegSL_1e-05_0.0001_0_0_0065_zmuv_fp.onnx")
            != std::string::npos){
        std::cout << "Using drn_d_22 unary config.\n";
        stixel_config_.segmentation_weight = 11.241965032069425f;
        stixel_config_.instance_weight = 0.0017313017435431333f;
        stixel_config_.disparity_weight = 0.0069935800364145494f;
        stixel_config_.eps = 23.89408062110343f;
        stixel_config_.min_pts = 4;
        stixel_config_.size_filter = 42;
    }
    else{
        std::cout << "Using drn_d_38 unary config.\n";
        stixel_config_.segmentation_weight = 14.94984454762259f;
        stixel_config_.instance_weight = 0.013686917379717443f;
        stixel_config_.disparity_weight = 0.0006375354572396317f;
        stixel_config_.eps = 18.54f;
        stixel_config_.min_pts = 4;
        stixel_config_.size_filter = 35;
    }
    onnx_filename_ = onnx_filename;

    trt_net_ =
        std::make_unique<TRTOnnxCNN>(
                onnx_filename_,
                input_tensornames_,
                output_tensornames_);
    // Loading ONNX model into TensorRT
    std::cout << "Building and running a GPU inference engine for onnx model\n";
    const bool successful_built = trt_net_->build();
    assert(successful_built);
}

StixelsWrapper::~StixelsWrapper(){
    if(stixels_.IsInitialized()){
        stixels_.Finish();
    }
    if(road_estimation_.IsInitialized()){
        road_estimation_.Finish();
    }
}

void StixelsWrapper::SetConfig(const StixelConfig& config){
    stixel_config_ = config;
    reinitialize_ = true;
}

StixelConfig StixelsWrapper::GetConfig(){
    return stixel_config_;
}


void StixelsWrapper::Reinitialize(){
    if(stixels_.IsInitialized()){
        stixels_.Finish();
    }
    stixels_.SetConfig(stixel_config_);
    stixels_.Initialize();

    if(road_estimation_.IsInitialized()){
        road_estimation_.Finish();
    }
    road_estimation_.Initialize(
            stixel_config_.camera_center_y, stixel_config_.baseline,
            stixel_config_.focal, stixel_config_.rows,
            stixel_config_.cols, stixel_config_.max_dis,
            stixel_config_.road_vdisparity_threshold);
    reinitialize_ = false;
}

std::vector<float> cvMatToVector(cv::Mat& img){
    std::vector<float> img_data(3*img.rows*img.cols);
    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            for(int channel = 0; channel < 3; channel++){
                int idx = channel *img.cols*img.rows
                          +row    *img.cols
                          +col;
                img_data[idx] =
                    ((unsigned int)
                        img.at<cv::Vec3b>(row, col)[channel])
                     / 255.0f;
            }
        }
    }
    return img_data;
}

void StixelsWrapper::ProcessFrame(cv::Mat& disparity_img, cv::Mat& rgb_img,
        std::unordered_map<std::string, float> camera_parameters){
    //TIME std::vector<float> times;

    // --- Read disparity image.
    if(stixel_config_.rows != disparity_img.rows
       || stixel_config_.cols != disparity_img.cols){
        stixel_config_.rows = disparity_img.rows;
        stixel_config_.cols = disparity_img.cols;
        reinitialize_ = true;
    }

    if (stixel_config_.baseline != camera_parameters["baseline"]
        || stixel_config_.focal != camera_parameters["focal"]
        || stixel_config_.camera_center_x
            != camera_parameters["center_x"]
        || stixel_config_.camera_center_y
            != camera_parameters["center_y"]) {

        stixel_config_.baseline = camera_parameters["baseline"];
        stixel_config_.focal = camera_parameters["focal"];
        stixel_config_.camera_center_x = camera_parameters["center_x"];
        stixel_config_.camera_center_y = camera_parameters["center_y"];

        std::cout << "New camera parameters: "
                  << "baseline = " << stixel_config_.baseline << ", "
                  << "focal = " << stixel_config_.focal << ", "
                  << "camera_center_y = "
                  << stixel_config_.camera_center_y << "\n";
    }

    if(reinitialize_) {
        Reinitialize();
    }
    //std::cout << "Initialized.\n";

    // TODO: Do the data juggling in some other way.
    std::vector<float> disparity_data(disparity_img.begin<float>(),
                                      disparity_img.end<float>());
    // Transfer disparity data to GPU.
    stixels_.SetDisparityImage(disparity_data);
    //std::cout << "Disparity set.\n";

    //TIME cudaDeviceSynchronize();
    //TIME std::chrono::steady_clock::time_point begin =
    //TIME         std::chrono::steady_clock::now();

    std::vector<float> rgb_data = cvMatToVector(rgb_img);
    // TensorRT inference
    // Keep data on device
    int32_t* infer_out = trt_net_->inferOnDevice(rgb_data);
    if (infer_out == nullptr){
        std::cout << "Inference failed!\n";
        return;
    }
    // Instead with a little detour over host memory:
    //~ auto infer_out = trt_net_->infer(rgb_data);
    //~ if (infer_out.size() == 0){
    //std::cout << "Inference done.\n";
    cudaDeviceSynchronize();
    // If you have a std vector.
    //~ stixels_.SetSegmentation(infer_out);
    //std::cout << "Segmentation set.\n";

    // --- Road estimation.
    // Transfering the disparity image to the GPU once more.
    //const bool ok = road_estimation_.Compute(disparity_data);
    const bool ok =
        road_estimation_.Compute(stixels_.GetInputDisparityImageOnDevice());
    if(!ok) {
        std::cout << "Can't compute road estimation.\n";
        return;
    }

    // Get and set road parameters.
    float camera_tilt = road_estimation_.GetPitch();
    float camera_height = road_estimation_.GetCameraHeight();
    float alpha_ground = road_estimation_.GetSlope();
    int vhor = road_estimation_.GetHorizonPoint();
    if(camera_tilt == 0 && camera_height == 0
            && vhor == 0 && alpha_ground == 0) {
        std::cout << "Can't compute road estimation.\n";
        return;
    }
    stixels_.SetRoadParameters(vhor, camera_tilt, camera_height,
                              alpha_ground);

    //std::cout << "Road estimated.\n";
    // Compute stixels.
    const float elapsed_time_ms =
        stixels_.Compute(stixel_config_.pairwise, stixels_data_, infer_out);
    cudaDeviceSynchronize();
    //std::cout << "Stixels computed.\n";
    //TIME std::chrono::steady_clock::time_point end =
    //TIME         std::chrono::steady_clock::now();
    //TIME int microseconds =
    //TIME     std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
    //TIME std::cout << "Done. Time elapsed (s): " << microseconds*1e-6 << "\n";

    //TIME if(first_time){
    //TIME     first_time = false;
    //TIME }
    //TIME else{ // skip the first time measurement as a warm up
    //TIME     //times.push_back(elapsed_time_ms);
    //TIME     times.push_back(microseconds*1e-3);
    //TIME }
}

std::vector<float>
StixelsWrapper::Get3DVertices(){
    return stixels_.Get3DVertices(stixels_data_);
}

const StixelsData&
StixelsWrapper::GetStixelsData(){
    return stixels_data_;
}

std::map<std::pair<int,int>, int>
StixelsWrapper::GetInstanceMapping(){
    return stixels_.GetInstanceStixels();
}
