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

// Description:
// This class is supposed to provide an interface between an application (e.g.
// the ROS node and the Stixels library. It also shows how to handle the
// configuration of the Stixels library and thus, together with the
// run_cityscapes.cu serves as an example of how to use the Stixels library.

#ifndef STIXELS_WRAPPER_H_
#define STIXELS_WRAPPER_H_

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

#include "Stixels.hpp" // Stixels
#include "RoadEstimation.h" // RoadEstimation
#include "TRTOnnxCNN.hpp"

class StixelsWrapper final {
public:
    StixelsWrapper(std::string onnx_filename);
    ~StixelsWrapper();

    void Reinitialize();
    void ProcessFrame(
            cv::Mat& disparity_img, cv::Mat& rgb_img,
            std::unordered_map<std::string, float> camera_parameters);
    const StixelsData& GetStixelsData();
    std::map<std::pair<int,int>, int> GetInstanceMapping();
    std::vector<float> Get3DVertices();
    void SetConfig(const StixelConfig& config);
    StixelConfig GetConfig();

private:
    StixelsData stixels_data_;
    StixelConfig stixel_config_;
    std::unique_ptr<TRTOnnxCNN> trt_net_;
    RoadEstimation road_estimation_;
    Stixels stixels_;
    bool reinitialize_ = true;

    std::string onnx_filename_;
    const std::vector<std::string> input_tensornames_{"input.1"};
    const std::vector<std::string> output_tensornames_{"output.1"};
};

#endif // STIXELS_WRAPPER_H_
