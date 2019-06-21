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

#include <iostream>
#include <fstream>
#include <dirent.h>
#include <string>
#include <sys/stat.h>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include "Stixels.hpp" // Stixels
#include "RoadEstimation.h" // RoadEstimation
#include "configuration.h" // pixel_t
#include "Segmentation.h" 

#define OVERWRITE   true

void SaveStixels(Section *stixels, const float alpha_ground, const int vhor,
                 const int real_cols, const int max_segments,
                 const char *fname) {
    std::ofstream fp;
    fp.open (fname, std::ofstream::out | std::ofstream::trunc);
    if(fp.is_open()) {
        for(size_t i = 0; i < real_cols; i++) {
            for(size_t j = 0; j < max_segments; j++) {
                Section section = stixels[i*max_segments+j];
                if(section.type == -1) {
                    break;
                }
                // If disparity is 0 it is sky
                if(section.type == OBJECT && section.disparity < 1.0f) {
                    section.type = SKY;
                }
                fp << section.type << "," << section.vB << "," << section.vT
                    << "," << section.disparity 
                    << "," << section.semantic_class
                    << "," << section.cost
                    << "," << section.instance_meanx
                    << "," << section.instance_meany << ";";
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

bool FileExists(const char *fname) {
    struct stat buffer;
    return (stat (fname, &buffer) == 0);
}

int main(int argc, char *argv[]) {
    if(argc < 7) {
        std::cerr << "Usage: stixels dir max_disparity segmentation_weight "
                  << "instance_weight disparity_weight stixel_width\n";
        return -1;
    }
    const char* directory = argv[1];
    const int max_dis = atoi(argv[2]);
    float segmentation_weight = atof(argv[3]);
    if(segmentation_weight < 1e-5)
        segmentation_weight = 1e-5;
    float instance_weight = atof(argv[4]) / segmentation_weight;
    if(instance_weight < 1e-8)
        instance_weight = 0.0;

    const float prior_weight = 1.0;
    const float disparity_weight = atof(argv[5]);
    const int column_step = atoi(argv[6]);
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
    char probs_file[PATH_MAX];
    char stixel_file[PATH_MAX];
    char camera_file[PATH_MAX];

    /* Parameters
     *
     */
    // --- Camera parameters.
    float focal = 0.0f;
    float baseline = 0.0f;
    float camera_center_y = 0.0f;

    /* Disparity Parameters */
    const float sigma_disparity_object = 1.0f;
    const float sigma_disparity_ground = 2.0f;
    const float sigma_sky = 0.1f; // Should be small compared to sigma_dis

    /* Probabilities */
    // Similar to values in Pfeiffer 14 dissertation, page 49.
    const float pout = 0.15f;
    const float pout_sky = 0.4f;
    const float pord = 0.2f;
    const float pgrav = 0.1f;
    const float pblg = 0.04f;

    //
    // 0.36, 0.3, 0.34 are similar to values in Pfeiffer 14 dissertation,
    // page 49.
    // However, unequal weighting did lead to invalid regions being classified as
    // ground or sky and instead of continuing an object.
    // Must add to 1.
    const float pground_given_nexist = 0.28;//1.f/3.;//0.36f;
    const float pobject_given_nexist = 0.44;//1.f/3.;//0.3;
    const float psky_given_nexist = 0.28;//1.f/3.;//0.34f;
    // tested: 0.2; 0.6; 0.2; but did not have significant effect.

    // Used this value from Pfeiffer 14 dissertation, page 49.
    const float pnexist_dis = 0.25f; // 0.0f;
    const float pground = 1.0f/3.0f;
    const float pobject = 1.0f/3.0f;
    const float psky = 1.0f/3.0f;
    // tested: 0.25; 0.5; 0.25; but did not have significant effect.

    /* Camera Paramters */
    int vhor;

    // Virtual parameters
    //const int column_step = 5;
    // Ignore a margin on the left side of the image.
    const int width_margin = 0;

    float camera_tilt;
    const float sigma_camera_tilt = 0.05f;
    float camera_height;
    const float sigma_camera_height = 0.05f;
    //const float camera_center_x = 651.216186523f;
    float alpha_ground;

    /* Model Parameters */
    const bool median_step = false;
    const float epsilon = 3.0f;
    const float range_objects_z = 10.20f; // in meters

    bool reinitialize = true;
    Stixels stixels;
    RoadEstimation road_estimation;
    std::vector<float> times;
    pixel_t *im;

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
        sprintf(probs_file, "%s/%s/%s_probs.h5", directory, probs_dir,
                base_filename.c_str());
        sprintf(stixel_file, "%s/%s/%s_%f.stixels", directory, stixel_dir,
                base_filename.c_str(), segmentation_weight);
        sprintf(camera_file, "%s/%s/%s_camera.json", directory, camera_dir,
                base_filename.c_str());

        if(!FileExists(stixel_file) || OVERWRITE) {
            cv::Mat dis = cv::imread(dis_file, cv::IMREAD_UNCHANGED);
            if(!dis.data) {
                std::cerr << "Couldn't read the file " << dis_file << std::endl;
                return EXIT_FAILURE;
            }

            // Convert images to grayscale
            if (dis.channels()>1) {
                cv::cvtColor(dis, dis, CV_RGB2GRAY);
            }

            Segmentation segmentation(probs_file);

            std::cout << ep->d_name << std::endl;
            std::cout << "Segmentation weight = " << segmentation_weight <<"\n";

            const int rows = dis.rows;
            const int cols = dis.cols;

            if(rows < max_dis) {
                printf("ERROR: Image height has to be equal or bigger than "
                        "maximum disparity\n");
                continue;
            }
            if(rows >= 1024) {
                printf("ERROR: Maximum image height has to be less than "
                        "1024\n");
                continue;
            }
            std::vector<hsize_t> probs_shape = segmentation.get_shape();
            if(rows != probs_shape[0]){
                std::cout << "ERROR: Height of disparity (" << rows
                        << ") and segmentation input (" << probs_shape[0]
                        << ") do not match.\n";
                continue;
            }
            if(cols != probs_shape[1]){
                std::cout << "ERROR: Width of disparity (" << cols
                        << ") and segmentation input (" << probs_shape[1]
                        << ") do not match.\n";
                continue;
            }

            // Load camera parameters from json file.
            if (FileExists(camera_file)) {
                std::cout << "File " << camera_file << " exists.\n";
                std::ifstream camera_ifs(camera_file);
                rapidjson::IStreamWrapper isw(camera_ifs);
                rapidjson::Document json_doc;
                json_doc.ParseStream(isw);
                float tmp_baseline =
                    json_doc["extrinsic"]["baseline"].GetDouble();
                float tmp_focal =
                    json_doc["intrinsic"]["fy"].GetDouble();
                float tmp_camera_center_y =
                    json_doc["intrinsic"]["v0"].GetDouble();

                if (baseline != tmp_baseline
                   || focal != tmp_focal
                   || camera_center_y != tmp_camera_center_y) {

                    baseline = tmp_baseline;
                    focal = tmp_focal;
                    camera_center_y = tmp_camera_center_y;

                    if (!reinitialize){
                        stixels.Finish();
                        road_estimation.Finish();
                        CUDA_CHECK_RETURN(cudaFreeHost(im));
                        reinitialize = true;
                    }
                    std::cout << "New camera parameters.\n";
                    std::cout << "baseline = " 
                              << baseline
                              << "\n";
                    std::cout << "focal = " 
                              << focal
                              << "\n";
                    std::cout << "camera_center_y = " 
                              << camera_center_y
                              << "\n";
                }
            }
            else {
                std::cout << "Error: Camera file " 
                          << camera_file << " does not exist.\n";
                return 1;
            }

            if(reinitialize) {
                stixels.SetDisparityParameters(rows, cols, max_dis, 
                                            sigma_disparity_object, 
                                            sigma_disparity_ground, sigma_sky);
                // channels = classes(=19) + regression(=2)
                stixels.SetSegmentationParameters(
                        19, segmentation.get_shape()[2]-19); // cityscapes
                stixels.SetWeightParameters(prior_weight, disparity_weight,
                                            segmentation_weight,
                                            instance_weight);
                stixels.SetProbabilities(pout, pout_sky, pground_given_nexist,
                                        pobject_given_nexist, psky_given_nexist,
                                        pnexist_dis, pground, pobject, psky,
                                        pord, pgrav, pblg);
                stixels.SetModelParameters(column_step, median_step, epsilon,
                                        range_objects_z, width_margin);
                stixels.SetCameraParameters(0.0f, focal, baseline, 0.0f,
                                        sigma_camera_tilt, 0.0f,
                                        sigma_camera_height, 0.0f);
                stixels.Initialize();
                road_estimation.Initialize(camera_center_y, baseline, focal,
                                        rows, cols, max_dis);

                CUDA_CHECK_RETURN(
                        cudaMallocHost((void**)&im,
                                       rows*cols*sizeof(pixel_t)));
                reinitialize = false;
            }
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

            // Transfer data to GPU.
            stixels.SetDisparityImage(im);
            stixels.SetSegmentation(segmentation.data());

            // Transfering the disparity image to the GPU once more.
            const bool ok = road_estimation.Compute(im);
            if(!ok) {
                printf("Can't compute road estimation\n");
                continue;
            }

            // Get Camera Parameters.
            camera_tilt = road_estimation.GetPitch();
            camera_height = road_estimation.GetCameraHeight();
            vhor = road_estimation.GetHorizonPoint();
            alpha_ground = road_estimation.GetSlope();

            if(camera_tilt == 0 && camera_height == 0 
                    && vhor == 0 && alpha_ground == 0) {
                printf("Can't compute road estimation\n");
                continue;
            }

            std::cout << "Camera Parameters -> Tilt: " << camera_tilt 
                    << " Height: " << camera_height << " vHor: " << vhor
                    << " alpha_ground: " << alpha_ground << "\n";

            stixels.SetCameraParameters(vhor, focal, baseline, camera_tilt,
                    sigma_camera_tilt, camera_height, sigma_camera_height, 
                    alpha_ground);

            std::cout << "Computing stixels.\n";
            const float elapsed_time_ms = stixels.Compute();
            std::cout << "Done.\n";

            times.push_back(elapsed_time_ms);

            Section *stx = stixels.GetStixels();

            // Saving vhor as rows-1-vhor. See GroundFunction in Stixels.cu
            // for comparison.
            SaveStixels(stx, alpha_ground, rows-1-vhor, stixels.GetRealCols(), 
                        stixels.GetMaxSections(), stixel_file);
            std::cout << "Finished.\n";
        }
    }
    if(!reinitialize) {
        stixels.Finish();
        road_estimation.Finish();
    }

    float mean = 0.0f;
    for(int i = 0; i < times.size(); i++) {
        mean += times.at(i);
    }
    mean = mean / times.size();
    std::cout << "It took an average of " << mean << " miliseconds, "
              << 1000.0f/mean << " fps" << std::endl;
    CUDA_CHECK_RETURN(cudaFreeHost(im));

    return 0;
}
