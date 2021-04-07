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

#include <string>
#include <map>
#include <vector>
#include <ros/ros.h>

// Dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <instance_stixels/InstanceStixelsConfig.h>

// Ros messages
#include <sensor_msgs/Image.h>
#include <stereo_msgs/DisparityImage.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>

#include "instance_stixels_msgs/InstanceStixelsArray.h"
#include "instance_stixels_msgs/InstanceStixel.h"

#include "stixels_wrapper.h"

using namespace sensor_msgs;
using namespace stereo_msgs;

class InstanceStixelsNode {
    private:
        typedef
            message_filters::sync_policies::ApproximateTime<
                DisparityImage, Image, CameraInfo, CameraInfo>
            StixelsNodeSyncPolicy;
        StixelsWrapper stixels_wrapper_;

        ros::Publisher pub_stixels_;
        ros::NodeHandle nh_;
        message_filters::Synchronizer<StixelsNodeSyncPolicy> sync_;
        message_filters::Subscriber<DisparityImage> sub_image_disp_;
        message_filters::Subscriber<Image> sub_image_left_;
        message_filters::Subscriber<CameraInfo> sub_info_left_;
        message_filters::Subscriber<CameraInfo> sub_info_right_;

        dynamic_reconfigure::Server<
            instance_stixels::InstanceStixelsConfig>
                cfg_server_;

        void populateStixelsArray(
            instance_stixels_msgs::InstanceStixelsArray& msg_stixels,
            const StixelsData& stixels_data,
            const std::map<std::pair<int,int>, int>& instance_mapping,
            const std::vector<float> vertices = {} );
    public:
        InstanceStixelsNode(ros::NodeHandle nh, std::string onnxfilename);
        //~InstanceStixelsNode() {};
        void callback(
            const DisparityImageConstPtr& image_disp_msg,
            const ImageConstPtr& image_msg,
            const CameraInfoConstPtr& l_info_msg,
            const CameraInfoConstPtr& r_info_msg);
        void reconfigure_callback(
            const instance_stixels::InstanceStixelsConfig& config,
            uint32_t level);
};
