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

// Ros messages
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/synchronizer.h>

#include <image_transport/image_transport.h>
#include <image_geometry/stereo_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "stixels_node.h"

using namespace sensor_msgs;
using namespace stereo_msgs;

std::string getImageType(int number) {
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt) {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type << "CV_" << imgTypeString << "C" << channel;

    return type.str();
}

void InstanceStixelsNode::populateStixelsArray(
        instance_stixels_msgs::InstanceStixelsArray& msg_stixels,
        const StixelsData& stixels_data,
        const std::map<std::pair<int,int>, int>& instance_mapping,
        const std::vector<float> vertices){
    msg_stixels.columns = stixels_data.realcols;
    msg_stixels.stixel_width = stixels_data.column_step;
    msg_stixels.image_height = stixels_data.rows;
    msg_stixels.semantic_classes = stixels_data.semantic_classes;
    msg_stixels.max_disparity = stixels_data.max_dis;

    auto vertex_iter = vertices.begin();
    int vertex_iter_step = 0;
    if(vertices.size() > 0){
        vertex_iter_step = 3 * 4; // xyz * 4 corners
        if(vertices.size() % vertex_iter_step != 0){
            vertex_iter_step = 0;
        }
    }
    std::vector<instance_stixels_msgs::InstanceStixel> stixels;
    for(int i = 0; i < stixels_data.realcols; i++) {
        for(int j = 0; j < stixels_data.max_sections; j++) {
            const Section& section =
                stixels_data.sections[i*stixels_data.max_sections+j];
            if(section.type == -1) {
                break;
            }

            const auto& it = instance_mapping.find(std::make_pair(i,j));
            const int instance_id =
                (it == instance_mapping.end()) ? -1 : it->second;

            instance_stixels_msgs::InstanceStixel stixel;
            stixel.column = i;
            stixel.type = (uint8_t) section.type;
            stixel.vB = section.vB;
            stixel.vT = section.vT;
            stixel.disparity = section.disparity;
            stixel.semantic_class = (uint8_t) section.semantic_class;
            stixel.instance_mean_u = section.instance_meanx;
            stixel.instance_mean_v = section.instance_meany;
            stixel.instance_id = instance_id;

            stixel.vertices.assign(vertex_iter, vertex_iter+vertex_iter_step);

            // This is probably resulting in a copy of "stixel" here. I guess
            // the compiler will take care of optimizing this one, but I don't
            // know for sure. Profile first, in case you're suspicous.
            stixels.push_back(stixel);
            vertex_iter += vertex_iter_step;
        }
    }
    msg_stixels.stixels = stixels;

    // Sanity check.
    if(vertices.size() != stixels.size() * vertex_iter_step){
        std::ostringstream err_str;
        err_str << "Number of vertices doesn't match number of stixels: "
                << stixels.size()
                << " stixels require "
                << stixels.size() * vertex_iter_step
                << " vertices, but got "
                << vertices.size() << " instead.\n";
        throw std::invalid_argument(err_str.str());
    }
}

void InstanceStixelsNode::callback(
        const DisparityImageConstPtr& image_disp_msg,
        const ImageConstPtr& image_msg,
        const CameraInfoConstPtr& l_info_msg,
        const CameraInfoConstPtr& r_info_msg) {
    // Read calibration info from camera info message.
    image_geometry::StereoCameraModel model;
    model.fromCameraInfo(*l_info_msg, *r_info_msg);
    std::unordered_map<std::string, float> camera_parameters;
    camera_parameters["baseline"] = (float) model.baseline();
    camera_parameters["focal"] = (float) model.left().fx(); // == fy() for ueye
    camera_parameters["center_x"] = (float) model.left().cx();
    camera_parameters["center_y"] = (float) model.left().cy();

    // Get images from messages.
    // TODO: use toCvShare (i.e. const ptr?)
    cv::Mat image_disp = cv_bridge::toCvCopy(image_disp_msg->image)->image;
    cv::Mat image = cv_bridge::toCvCopy(image_msg, "rgb8")->image;
    //cvtColor(image, image, CV_BGR2RGB);

    // Resize/crop both images.
    constexpr int crop_width = 1792;
    constexpr int crop_height = 784;
    int offset_x = (image_disp.cols - crop_width) / 2;
    int offset_y = (image_disp.rows - crop_height) / 2;

    // We would like to avoid to cut off too much from the bottom, but rather cut off sky
    // parts.
    const int max_bottom_offset = 20;
    if(offset_y > max_bottom_offset){
        offset_y = image_disp.rows - 784 - max_bottom_offset;
    }

    cv::Rect valid_roi(offset_x, offset_y, crop_width, crop_height);
    image_disp = image_disp(valid_roi);
    image      =      image(valid_roi);

    // Correct camera parameters.
    // I expect that these coordinates are measured from the top left of the image.
    camera_parameters["center_y"] -= offset_y;
    camera_parameters["center_x"] -= offset_x;

    ROS_DEBUG("disparity_image = %d x %d x %d (type %s)\n",
              image_disp.cols, image_disp.rows, image_disp.channels(),
              getImageType(image_disp.type()).c_str());
    ROS_DEBUG("image = %d x %d x %d (type %s)\n",
              image.cols, image.rows, image.channels(),
              getImageType(image.type()).c_str());

    ROS_DEBUG( "Processing frame.");
    stixels_wrapper_.ProcessFrame(image_disp, image, camera_parameters);
    ROS_DEBUG( "Done processing frame.");

    // Populate StixelsArray message
    instance_stixels_msgs::InstanceStixelsArray msg_stixels;
    msg_stixels.header = image_msg->header;

    sensor_msgs::RegionOfInterest msg_valid_roi;
    msg_valid_roi.height = valid_roi.height;
    msg_valid_roi.width = valid_roi.width;
    msg_valid_roi.x_offset = valid_roi.x;
    msg_valid_roi.y_offset = valid_roi.y;
    msg_valid_roi.do_rectify = false;
    msg_stixels.valid_region = msg_valid_roi;

    auto stixels_data = stixels_wrapper_.GetStixelsData();
    auto instance_mapping = stixels_wrapper_.GetInstanceMapping();
    auto vertices = stixels_wrapper_.Get3DVertices();
    populateStixelsArray(
            msg_stixels, stixels_data, instance_mapping, vertices);
    //std::cout << "stixels.size() = " << msg_stixels.stixels.size() << "\n";

    pub_stixels_.publish(msg_stixels);
}

void InstanceStixelsNode::reconfigure_callback(
        const instance_stixels::InstanceStixelsConfig& config,
        uint32_t level) {
    auto stixel_config = stixels_wrapper_.GetConfig();

    // Weights
    stixel_config.disparity_weight = config.disparity_weight;
    stixel_config.segmentation_weight = config.segmentation_weight;
    stixel_config.instance_weight = config.instance_weight;
    stixel_config.prior_weight = config.prior_weight;
    stixel_config.pairwise = config.pairwise;

    // Ground parameters
    stixel_config.sigma_disparity_ground = config.sigma_disparity_ground;

    // Instance clustering parameters
    stixel_config.eps = config.eps;
    stixel_config.min_pts = config.min_pts;
    stixel_config.size_filter = config.size_filter;

    stixel_config.invalid_disparity = (config.invalid_disparity ? 0.0f : -1.0f);

    stixel_config.sigma_disparity_object = config.sigma_disparity_object;
    stixel_config.sigma_disparity_ground = config.sigma_disparity_ground;
    stixel_config.sigma_sky = config.sigma_sky; // Should be small compared to sigma_dis

    /* Probabilities */
    // Similar to values in Pfeiffer 14 dissertation, page 49.
    stixel_config.pout = config.pout;
    stixel_config.pout_sky = config.pout_sky;
    stixel_config.pord = config.pord;
    stixel_config.pgrav = config.pgrav;
    stixel_config.pblg = config.pblg;

    // 0.36, 0.3, 0.34 are similar to values in Pfeiffer 14 dissertation,
    // page 49.
    // However, unequal weighting did lead to invalid regions being classified as
    // ground or sky and instead of continuing an object.
    // Must add to 1.
    stixel_config.pground_given_nexist = config.pground_given_nexist;
    stixel_config.pobject_given_nexist = config.pobject_given_nexist;
    stixel_config.psky_given_nexist = config.psky_given_nexist;
    // tested: 0.2; 0.6; 0.2; but did not have significant effect.

    // Used this value from Pfeiffer 14 dissertation, page 49.
    stixel_config.pnexist_dis = config.pnexist_dis;
    stixel_config.pground = config.pground;
    stixel_config.pobject = config.pobject;
    stixel_config.psky = config.psky;
    // tested: 0.25; 0.5; 0.25; but did not have significant effect.

    stixel_config.sigma_camera_tilt = config.sigma_camera_tilt;
    stixel_config.sigma_camera_height = config.sigma_camera_height;
    //const stixel_config.camera_center_x = config.camera_center_x;

    /* Model Parameters */
    stixel_config.median_join = config.median_join;
    stixel_config.epsilon = config.epsilon;
    stixel_config.range_objects_z = config.range_objects_z; // in meters

    stixel_config.road_vdisparity_threshold = config.road_vdisparity_threshold;

    stixels_wrapper_.SetConfig(stixel_config);
}

InstanceStixelsNode::InstanceStixelsNode(
        ros::NodeHandle nh, std::string onnxfilename) 
        : 
        nh_(nh), sync_(StixelsNodeSyncPolicy(10)),
        stixels_wrapper_(onnxfilename) {
    // subscribe
    sub_image_disp_.subscribe(nh_,"disparity", 1);
    sub_image_left_.subscribe(nh_,"left/image_color", 1);
    sub_info_left_.subscribe(nh_,"left/camera_info", 1);
    sub_info_right_.subscribe(nh_,"right/camera_info", 1);

    sync_.connectInput(
            sub_image_disp_, sub_image_left_, sub_info_left_, sub_info_right_);
    sync_.registerCallback(
            boost::bind(&InstanceStixelsNode::callback, this, _1, _2, _3, _4));

    //// publish
    pub_stixels_ =
        nh_.advertise<instance_stixels_msgs::InstanceStixelsArray>(
                "instance_stixels/stixels", 100);

    // setup dynamic reconfigure server
    dynamic_reconfigure::Server<
        instance_stixels::InstanceStixelsConfig>::CallbackType f;
    cfg_server_.setCallback(boost::bind(
                &InstanceStixelsNode::reconfigure_callback, this, _1, _2));
}
