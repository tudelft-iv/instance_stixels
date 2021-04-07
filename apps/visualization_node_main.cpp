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

#include <ros/ros.h>
#include "visualization_node.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "instance_stixels_visualization");

    ros::NodeHandle nh;

    InstanceStixelsVisualizationNode visualization_node(nh);

    ROS_INFO("Spinning stixel visualization...");
    ros::spin();
    ROS_INFO("InstanceStixels done...");

    return 0;
}

