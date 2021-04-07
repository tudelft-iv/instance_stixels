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

#include <opencv2/opencv.hpp>
#include <ros/ros.h>

// Dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <instance_stixels/InstanceStixelsVisualizationConfig.h>

// Ros messages
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <stereo_msgs/DisparityImage.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "instance_stixels_msgs/InstanceStixel.h"
#include "instance_stixels_msgs/InstanceStixelsArray.h"

using namespace sensor_msgs;
using namespace stereo_msgs;

class InstanceStixelsVisualizationNode {
    private:
        enum class ColorScheme {
            DISPARITY,
            SEMANTIC,
            INSTANCE,
        };

        typedef
            message_filters::sync_policies::ApproximateTime<
                instance_stixels_msgs::InstanceStixelsArray, Image>
            VisualizationNodeSyncPolicy;

        static const std::vector<std::string> CLASSNAMES_CITYSCAPES;

        std::array<bool, 19> add_class_to_pointcloud_ =
              { false, false, // road and sidewalk
                true, true, true, true, true, true, true, true,
                false, // sky
                true, true, true, true, true, true, true, true };
        float image_overlay_alpha_ = 0.7;
        ColorScheme image_color_scheme_
            = ColorScheme::SEMANTIC;

        // ROS
        ros::NodeHandle nh_;
        image_transport::Publisher pub_image_stixel_;
        ros::Publisher pub_pointcloud_;
        ros::Publisher pub_markerarray_;
        message_filters::Synchronizer<VisualizationNodeSyncPolicy> sync_;
        message_filters::Subscriber<Image> sub_image_left_;
        message_filters::Subscriber<instance_stixels_msgs::InstanceStixelsArray>
            sub_stixels_;
        sensor_msgs::PointField x_field_, y_field_, z_field_;

        dynamic_reconfigure::Server<
            instance_stixels::InstanceStixelsVisualizationConfig>
                cfg_server_;

        // methods
        void populateSemanticMarkerArray(
                visualization_msgs::MarkerArray& msg_markerarray,
                const instance_stixels_msgs::InstanceStixelsArrayConstPtr& stixels_msg);
        cv::Mat DrawStixels(
                const instance_stixels_msgs::InstanceStixelsArray& stixels_msg,
                const ColorScheme color_scheme);
        cv::Scalar DisparityToColor(
                const instance_stixels_msgs::InstanceStixel& stixel);
        cv::Scalar SemanticClassToColor(
                const instance_stixels_msgs::InstanceStixel& stixel);
        cv::Scalar SemanticClassToColor(
                const int semantic_class);
        cv::Scalar InstanceToColor(
                const instance_stixels_msgs::InstanceStixel& stixel);
     public:
        InstanceStixelsVisualizationNode(ros::NodeHandle nh);
        void callback(
                const instance_stixels_msgs::InstanceStixelsArrayConstPtr& stixels_msg,
                const ImageConstPtr& image_msg);
        void reconfigure_callback(
            const instance_stixels::InstanceStixelsVisualizationConfig&
                config,
            uint32_t level);
};
