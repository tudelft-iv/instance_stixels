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

#include <exception>
#include <iostream>
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <tuple>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include "Stixels.hpp" // Stixels
#include "RoadEstimation.h" // RoadEstimation
#include "configuration.h" // pixel_t
// TODO: remove this dependency. pixel_t is not going to change anyway
#include "H5Segmentation.h"
#include "TRTOnnxCNN.hpp"

constexpr bool OVERWRITE = true;

bool FileExists(const char *fname) {
    struct stat buffer;
    return (stat (fname, &buffer) == 0);
}

std::unordered_map<std::string, float>
LoadCameraFile(const char* camera_file){
    std::unordered_map<std::string, float> camera_parameters;
    if (FileExists(camera_file)) {
        std::cout << "File " << camera_file << " exists.\n";
        std::ifstream camera_ifs(camera_file);
        rapidjson::IStreamWrapper isw(camera_ifs);
        rapidjson::Document json_doc;
        json_doc.ParseStream(isw);
        camera_parameters["baseline"] =
            json_doc["extrinsic"]["baseline"].GetDouble();
        camera_parameters["focal"] =
            json_doc["intrinsic"]["fy"].GetDouble();
        camera_parameters["center_y"] =
            json_doc["intrinsic"]["v0"].GetDouble();
    }
    else{
        std::cout << "Warning: Camera file "
                  << camera_file
                  << " does not exist. Falling back to UEYE parameters!\n";
        //return 1;
        //  //UEYE  (resized to 1624x1020 -- downscale factor ~ 0.8388)
        const float size_factor = 1000./1216.;
        camera_parameters["focal"] = 1495.46f; // * size_factor;           //1254.399f;
        camera_parameters["baseline"] = 0.22087f;
        camera_parameters["center_y"] = 624.896 * size_factor;     //524.163f;
    }
    return camera_parameters;
}

std::vector<float>
readRGBImage(const char* img_file){
    cv::Mat img = cv::imread(img_file, cv::IMREAD_UNCHANGED);
    cv::cvtColor(img, img, CV_BGR2RGB);
    if(!img.data) {
        std::ostringstream err_str;
        err_str << "Couldn't read the file " << img_file;
        throw std::invalid_argument(err_str.str());
    }
    // I am sure there are better ways to copy the data into the
    // vector, but it's fine for now.
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

std::tuple<std::vector<pixel_t>, int, int>
readDisparityImage(const char* dis_file, const int max_dis){
    cv::Mat dis = cv::imread(dis_file, cv::IMREAD_UNCHANGED);
    if(!dis.data) {
        std::ostringstream err_str;
        err_str << "Couldn't read the file " << dis_file;
        throw std::invalid_argument(err_str.str());
    }
    if (dis.channels()>1) {
        cv::cvtColor(dis, dis, CV_RGB2GRAY);
    }
    const int rows = dis.rows;
    const int cols = dis.cols;

    if(rows < max_dis) {
        std::ostringstream err_str;
        err_str << "ERROR: Image height has to be equal or bigger than "
                   "maximum disparity.";
        throw std::invalid_argument(err_str.str());
    }
    if(rows >= 1024) {
        std::ostringstream err_str;
        err_str << "ERROR: Maximum image height has to be less than "
                   "1024.";
        throw std::invalid_argument(err_str.str());
    }
    std::vector<pixel_t> im(rows*cols);
    if(dis.depth() == CV_8U) {
        for(int i = 0; i < dis.rows; i++) {
            for(int j = 0; j < dis.cols; j++) {
                const pixel_t d = (float) dis.at<uint8_t>(i, j);
                im[i*dis.cols+j] = d;
            }
        }
    } else {
        for(int i = 0; i < dis.rows; i++) {
            for(int j = 0; j < dis.cols; j++) {
                const pixel_t d = (float) dis.at<uint16_t>(i, j)/256.0f;
                im[i*dis.cols+j] = d;
            }
        }
    }
    return std::make_tuple(im, rows, cols);
}


// run tests/data//short_test/left_org/.. 128 14.94984454762259 0.013686917379717443 0.0006375354572396317 0 8 18.54 4 35
int main(int argc, char *argv[]) {
    // --- Parse arguments.
    if(argc < 11) {
        std::cerr << "Usage: stixels dir max_disparity segmentation_weight "
                  << "instance_weight disparity_weight pairwise stixel_width "
                  << "eps min_pts size_filter\n";
        return -1;
    }
    const char* directory = argv[1];
    const int max_dis = atoi(argv[2]);
    const float segmentation_weight = atof(argv[3]);
    const float instance_weight = atof(argv[4]);

    const float disparity_weight = atof(argv[5]);
    const bool pairwise = atoi(argv[6]);
    const float prior_weight = pairwise ? 1 : 1e4;

    const int column_step = atoi(argv[7]);
    const float eps = atof(argv[8]);
    const int min_pts = atoi(argv[9]);
    const int size_filter = atoi(argv[10]);
    bool use_tensorrt = false;
    if(argc > 11){
        if(atoi(argv[11]) == 1){
            use_tensorrt = true;
        }
    }

    // --- Setup stixel configuration.
    StixelConfig stixel_config;
    stixel_config.column_step = column_step;
    stixel_config.max_dis = max_dis;
    stixel_config.invalid_disparity = 0.0f;
    stixel_config.n_semantic_classes = 19; // cityscapes
    stixel_config.n_offset_channels = 2;
    stixel_config.prior_weight = prior_weight;
    stixel_config.segmentation_weight = segmentation_weight;
    stixel_config.instance_weight = instance_weight;
    stixel_config.disparity_weight = disparity_weight;
    stixel_config.eps = eps;
    stixel_config.min_pts = min_pts;
    stixel_config.size_filter = size_filter;

    // --- Setup directory and filename structures.
    const char* disparity_dir = "disparities";
    const char* probs_dir = "probs";
    const char* stixel_dir = "stixels";
    const char* camera_dir = "camera";

    DIR *dp;
    struct dirent *ep;
    char abs_dis_dir[PATH_MAX];
    sprintf(abs_dis_dir, "%s/%s", directory, disparity_dir);
    dp = opendir(abs_dis_dir);
    if (dp == NULL) {
        std::cerr << "Invalid directory: " << abs_dis_dir << std::endl;
        exit(EXIT_FAILURE);
    }
    char dis_file[PATH_MAX];
    char img_file[PATH_MAX];
    char probs_file[PATH_MAX];
    char stixel_file[PATH_MAX];
    char camera_file[PATH_MAX];

    // --- Frame dependent road parameters.
    int vhorizon_point; // Mirrored to vhor in Stixels.
    float camera_tilt;
    float camera_height;
    float alpha_ground;

    // --- Setup TensorRT.
    std::string onnxFileName =
        "weights/onnx/DRNDSDoubleSegSL_0.0001_0.0001_0_0_0095_zmuv_fp.onnx";
    std::vector<std::string> inputTensorNames{"input.1"};
    std::vector<std::string> outputTensorNames{"output.1"};
    TRTOnnxCNN trt_net(onnxFileName, inputTensorNames, outputTensorNames);
    if(use_tensorrt){
        // Loading ONNX model into TensorRT
        std::cout << "Building and running a GPU inference engine for onnx model\n";
        if (!trt_net.build()){
            std::cout << "Builiding failed!\n";
            return false;
        }
    }

    // --- Iterate over files in disparity directory.
    bool first_time = true; // Skip first time measurement for warm-up.
    bool reinitialize = true;
    StixelsData stixels_data;
    Stixels stixels;
    RoadEstimation road_estimation;
    std::vector<float> times;

    while ((ep = readdir(dp)) != NULL) {
        if (!strcmp (ep->d_name, "."))
            continue;
        if (!strcmp (ep->d_name, ".."))
            continue;
        std::string dis_filename(ep->d_name);
        std::string base_filename(dis_filename);
        base_filename.erase(base_filename.find("_disparity.png"), 14);

        sprintf(dis_file, "%s/%s/%s", directory, disparity_dir,
                dis_filename.c_str());
        sprintf(img_file, "%s/%s/../left/%s_leftImg8bit.png", directory, disparity_dir,
                base_filename.c_str());
        sprintf(probs_file, "%s/%s/%s_probs.h5", directory, probs_dir,
                base_filename.c_str());
        sprintf(stixel_file, "%s/%s/%s.stixels", directory, stixel_dir,
                base_filename.c_str());
        sprintf(camera_file, "%s/%s/%s_camera.json", directory, camera_dir,
                base_filename.c_str());

        if(!FileExists(stixel_file) || OVERWRITE) {
            std::cout << ep->d_name << std::endl;

            // --- Read disparity image.
            std::tuple<std::vector<pixel_t>, int, int> disparity_img_tuple;
            try {
                disparity_img_tuple = readDisparityImage(dis_file, max_dis);
            }
            catch (std::invalid_argument err){
                std::cerr << err.what() << "\n";
                continue;
            }
            const auto& disparity_img = std::get<0>(disparity_img_tuple);
            const int current_rows = std::get<1>(disparity_img_tuple);
            const int current_cols = std::get<2>(disparity_img_tuple);
            if(stixel_config.rows != current_rows
               || stixel_config.cols != current_cols){
                stixel_config.rows = current_rows;
                stixel_config.cols = current_cols;
                reinitialize = true;
            }

            // --- Read RGB image.
            std::vector<float> img_data;
            if(use_tensorrt){
                try{
                    img_data = readRGBImage(img_file);
                }
                catch (std::invalid_argument err){
                    std::cerr << err.what() << "\n";
                    continue;
                }
                if(img_data.size() != stixel_config.rows*stixel_config.cols*3){
                    std::cerr << "RGB and disparity image dimensions do not "
                              << "match.\n";
                    continue;
                }
            }

            // Load and check camera parameters from json file.
            std::unordered_map<std::string, float> camera_parameters =
                LoadCameraFile(camera_file);
            if (stixel_config.baseline != camera_parameters["baseline"]
                || stixel_config.focal != camera_parameters["focal"]
                || stixel_config.camera_center_y
                    != camera_parameters["center_y"]) {

                stixel_config.baseline = camera_parameters["baseline"];
                stixel_config.focal = camera_parameters["focal"];
                stixel_config.camera_center_y = camera_parameters["center_y"];

                reinitialize = true;
                std::cout << "New camera parameters: "
                          << "baseline = " << stixel_config.baseline << ", "
                          << "focal = " << stixel_config.focal << ", "
                          << "camera_center_y = "
                          << stixel_config.camera_center_y << "\n";
            }

            if(reinitialize) {
                if(stixels.IsInitialized()){
                    stixels.Finish();
                }
                if(road_estimation.IsInitialized()){
                    road_estimation.Finish();
                }
                stixels.SetConfig(stixel_config);
                stixels.Initialize();
                road_estimation.Initialize(
                        stixel_config.camera_center_y, stixel_config.baseline,
                        stixel_config.focal, stixel_config.rows,
                        stixel_config.cols, stixel_config.max_dis);

                reinitialize = false;
            }

            // Transfer disparity data to GPU.
            stixels.SetDisparityImage(disparity_img);

            H5Segmentation segmentation;
            if(!use_tensorrt){ // Loading does not need to be timed.
                segmentation.LoadFile(probs_file);

                std::vector<hsize_t> probs_shape = segmentation.get_shape();
                if(pow(2, ceil(log2(stixel_config.rows / 8 +1))) != probs_shape[2]){
                    std::cout << "ERROR: Height of disparity ("
                              << stixel_config.rows
                              << ") and segmentation input (" << probs_shape[2]
                              << ") do not match. "
                              << "Segmentation input should be "
                              << pow(2, ceil(log2(stixel_config.rows / 8 + 1)))
                              << ".\n";
                    continue;
                }
                if(stixel_config.cols / 8 != probs_shape[0]){
                    std::cout << "ERROR: Width of disparity ("
                              << stixel_config.cols
                              << ") and segmentation input (" << probs_shape[0]
                              << ") do not match.\n";
                    continue;
                }
            }

            cudaDeviceSynchronize();
            std::chrono::steady_clock::time_point begin =
                    std::chrono::steady_clock::now();

            if(use_tensorrt){
                // TensorRT inference
                auto infer_out = trt_net.infer(img_data);
                if (infer_out.size() == 0){
                    std::cout << "Inference failed!\n";
                    return false;
                }
                stixels.SetSegmentation(infer_out);
            }
            else{
                stixels.SetSegmentation(segmentation.data());
            }

            // Transfering the disparity image to the GPU once more.
            const bool ok = road_estimation.Compute(disparity_img);
            if(!ok) {
                printf("Road estimation failed.\n");
                continue;
            }

            // Get and set road parameters.
            camera_tilt = road_estimation.GetPitch();
            camera_height = road_estimation.GetCameraHeight();
            vhorizon_point = road_estimation.GetHorizonPoint();
            alpha_ground = road_estimation.GetSlope();
            if(camera_tilt == 0 && camera_height == 0
                    && vhorizon_point == 0 && alpha_ground == 0) {
                printf("Invalid road estimation.\n");
                continue;
            }
            stixels.SetRoadParameters(vhorizon_point, camera_tilt, camera_height,
                                      alpha_ground);

            // Compute stixels.
            const float elapsed_time_ms =
                stixels.Compute(pairwise, stixels_data);
            cudaDeviceSynchronize();
            std::chrono::steady_clock::time_point end =
                    std::chrono::steady_clock::now();
            int microseconds =
                std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();

            std::cout << "Done. Time elapsed (s): " << microseconds*1e-6 << "\n";

            if(first_time){
                first_time = false;
            }
            else{ // skip the first time measurement as a warm up
                //times.push_back(elapsed_time_ms);
                times.push_back(microseconds*1e-3);
            }

            // Get and save stixels.
            Section *stx = stixels_data.sections.data();
            std::map<std::pair<int,int>, int> instance_stixels =
                stixels.GetInstanceStixels();
            for(const auto& it : instance_stixels){
                int i = it.first.first;
                int j = it.first.second;
                int class_id = stx[i*stixels.GetMaxSections()+j].semantic_class;
                if(class_id < 11){
                    std::cout << "(" << it.first.first
                              << "," << it.first.second
                              << "):" << it.second;
                    std::cout << " and class id " << class_id;
                    std::cout << "\n";
                }
            }

            // Saving vhor as rows-1-vhorizon_point. See GroundFunction in
            // Stixels.cu for comparison.
            Stixels::SaveStixels(stx, instance_stixels, alpha_ground,
                    stixel_config.rows-1-vhorizon_point, stixels.GetRealCols(),
                    stixels.GetMaxSections(), stixel_file);
            std::cout << "Finished.\n";
        }
    }
    float mean = 0.0f;
    for(int i = 0; i < times.size(); i++) {
        mean += times.at(i);
    }
    mean = mean / times.size();
    std::cout << "It took an average of " << mean << " milliseconds, "
              << 1000.0f/mean << " fps" << std::endl;

    // Free memory.
    // TODO: Ideally, this would be done by the destructor.
    if(stixels.IsInitialized()){
        stixels.Finish();
    }
    if(road_estimation.IsInitialized()){
        road_estimation.Finish();
    }

    return 0;
}
