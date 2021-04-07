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

#include <cv_bridge/cv_bridge.h>

#include "visualization_node.h"

using namespace sensor_msgs;
using namespace stereo_msgs;

const std::vector<std::string>
    InstanceStixelsVisualizationNode::CLASSNAMES_CITYSCAPES = {
        {"road"}, {"sidewalk"}, {"building"},
        {"wall"}, {"fence"}, {"pole"},
        {"traffic light"}, {"traffic sign"}, {"vegetation"},
        {"terrain"}, {"sky"}, {"person"},
        {"rider"}, {"car"}, {"truck"},
        {"bus"}, {"train"}, {"motocycle"},
        {"bicycle"}
   };

cv::Scalar InstanceStixelsVisualizationNode::InstanceToColor(
        const instance_stixels_msgs::InstanceStixel& stixel) {
    // 30 arbitrary innstance colors.
    // Visually distinct palette based on: https://mokole.com/palette.html
    static const std::vector<cv::Scalar> COLORS_INSTANCES = {
            {0, 0, 0}, {105, 105, 105}, {85, 107, 47},
            {34, 139, 34}, {127, 0, 0}, {72, 61, 139},
            {0, 139, 139}, {70, 130, 180}, {210, 105, 30},
            {0, 0, 139}, {50, 205, 50}, {218, 165, 32},
            {143, 188, 143}, {139, 0, 139}, {255, 69, 0},
            {255, 255, 0}, {0, 255, 0}, {0, 250, 154},
            {138, 43, 226}, {233, 150, 122}, {220, 20, 60},
            {0, 255, 255}, {0, 0, 255}, {216, 191, 216},
            {255, 0, 255}, {30, 144, 255}, {219, 112, 147},
            {255, 20, 147}, {238, 130, 238}, {255, 228, 181}
        };

    cv::Scalar color(255,255,255);
    if( stixel.instance_id != -1 ){
        const int color_idx =
            (stixel.semantic_class * (stixel.instance_id+1))
            % COLORS_INSTANCES.size();
        color = COLORS_INSTANCES[color_idx];
    }
    return color;
}

cv::Scalar InstanceStixelsVisualizationNode::DisparityToColor(
        const instance_stixels_msgs::InstanceStixel& stixel) {
    static const cv::Scalar COLOR_SKY(255, 100, 100);
    static const cv::Scalar COLOR_GROUND(255, 255, 255);

    if(stixel.type == instance_stixels_msgs::InstanceStixel::OBJECT){
        // Disparity as color gradient
        float hue = 0.6 * ((70.-stixel.disparity)/70.);
        hue = (hue < 0.) ? 0. : hue;
        hue = (hue > 1.) ? 1. : hue;
        return cv::Scalar(100, hue*255, 255-hue*255);
    }
    else if(stixel.type == instance_stixels_msgs::InstanceStixel::SKY){
        return COLOR_SKY;
    }
    return COLOR_GROUND;
}

cv::Scalar InstanceStixelsVisualizationNode::SemanticClassToColor(
        const instance_stixels_msgs::InstanceStixel& stixel) {
    return SemanticClassToColor(stixel.semantic_class);
}

cv::Scalar InstanceStixelsVisualizationNode::SemanticClassToColor(
        const int semantic_class) {
    static const std::vector<cv::Scalar> COLOR_CITYSCAPES = {
            {128, 64, 128}, {244, 35, 232}, {70, 70, 70},
            // 0 = road, 1 = sidewalk, 2 = building
            {102, 102, 156}, {190, 153, 153}, {153, 153, 153},
            // 3 = wall, 4 = fence, 5 = pole
            {250, 170, 30}, {220, 220, 0}, {107, 142, 35},
            // 6 = traffic light, 7 = traffic sign, 8 = vegetation
            {152, 251, 152}, {70, 130, 180}, {220, 20, 60},
            // 9 = terrain, 10 = sky, 11 = person
            {255, 0, 0}, {0, 0, 142}, {0, 0, 70},
            // 12 = rider, 13 = car, 14 = truck
            {0, 60, 100}, {0, 80, 100}, {0, 0, 230},
            // 15 = bus, 16 = train, 17 = motocycle
            {119, 11, 32}
            // 18 = bicycle
       };
    return COLOR_CITYSCAPES[semantic_class];
}

cv::Mat InstanceStixelsVisualizationNode::DrawStixels(
        const instance_stixels_msgs::InstanceStixelsArray& stixels_msg,
        const ColorScheme color_scheme) {
    static const cv::Scalar COLOR_OUTLINE(10, 10, 10);

    cv::Mat out(stixels_msg.image_height,
                stixels_msg.columns * stixels_msg.stixel_width,
                CV_8UC3);
    out.setTo(cv::Scalar(0,0,0));

    cv::Point pt1, pt2;

    for(const auto& stixel : stixels_msg.stixels){
        pt1.x = stixel.column * stixels_msg.stixel_width;
        pt1.y = stixels_msg.image_height - stixel.vB; // bottom
        pt2.x = pt1.x + stixels_msg.stixel_width - 1;
        pt2.y = stixels_msg.image_height - stixel.vT - 1; // top

        cv::Scalar color_fill;
        switch(color_scheme){
            case (ColorScheme::SEMANTIC):
                color_fill = SemanticClassToColor(stixel);
                break;
            case (ColorScheme::INSTANCE):
                color_fill = InstanceToColor(stixel);
                break;
            default: // ColorScheme::DISPARITY
                color_fill = DisparityToColor(stixel);
                break;
        }

        cv::rectangle(out, pt1, pt2, color_fill, CV_FILLED);
        cv::rectangle(out, pt1, pt2, COLOR_OUTLINE, 1); // outline
    }

    return out;
}

void InstanceStixelsVisualizationNode::populateSemanticMarkerArray(
        visualization_msgs::MarkerArray& msg_markerarray,
        const instance_stixels_msgs::InstanceStixelsArrayConstPtr& stixels_msg){
    constexpr int CITYSCAPES_CLASSES = 19;
    std::vector<visualization_msgs::Marker> marker_msgs(CITYSCAPES_CLASSES);
    for(int class_i = 0; class_i < CITYSCAPES_CLASSES; class_i++){
        marker_msgs[class_i].header = stixels_msg->header;
        marker_msgs[class_i].ns = CLASSNAMES_CITYSCAPES[class_i];
        marker_msgs[class_i].id = class_i;
        marker_msgs[class_i].type = visualization_msgs::Marker::TRIANGLE_LIST;
        marker_msgs[class_i].action = visualization_msgs::Marker::ADD;
        marker_msgs[class_i].lifetime = ros::Duration(0.0);
        marker_msgs[class_i].frame_locked = true;

        marker_msgs[class_i].pose.orientation.x = 0.0;
        marker_msgs[class_i].pose.orientation.y = 0.0;
        marker_msgs[class_i].pose.orientation.z = 0.0;
        marker_msgs[class_i].pose.orientation.w = 1.0;

        marker_msgs[class_i].scale.x = 1.0;
        marker_msgs[class_i].scale.y = 1.0;
        marker_msgs[class_i].scale.z = 1.0;

        const auto& class_color = SemanticClassToColor(class_i);
        marker_msgs[class_i].color.r = class_color[0] / 255.0f;
        marker_msgs[class_i].color.g = class_color[1] / 255.0f;
        marker_msgs[class_i].color.b = class_color[2] / 255.0f;
        marker_msgs[class_i].color.a = 1.0;

        // Add a dummy triangle so that namespace shows up in rviz.
        geometry_msgs::Point p;
        p.x = 0;
        p.y = 0;
        p.z = 0;
        marker_msgs[class_i].points.push_back(p);
        marker_msgs[class_i].points.push_back(p);
        marker_msgs[class_i].points.push_back(p);
    }
    // Fill point lists.
    for(const auto& stixel_msg : stixels_msg->stixels){
        if(stixel_msg.type != instance_stixels_msgs::InstanceStixel::SKY){
            const auto& vertices = stixel_msg.vertices;
            // vertices 0-2 top left, 3-5 top right, ..., 9-11 bottom left
            geometry_msgs::Point top_left;
            top_left.x = vertices[0];
            top_left.y = vertices[1];
            top_left.z = vertices[2];

            geometry_msgs::Point top_right;
            top_right.x = vertices[3];
            top_right.y = vertices[4];
            top_right.z = vertices[5];

            geometry_msgs::Point bottom_right;
            bottom_right.x = vertices[6];
            bottom_right.y = vertices[7];
            bottom_right.z = vertices[8];

            geometry_msgs::Point bottom_left;
            bottom_left.x = vertices[9];
            bottom_left.y = vertices[10];
            bottom_left.z = vertices[11];

            // Upper left triangle of the stixel's quadrilateral.
            marker_msgs[stixel_msg.semantic_class].points.push_back(bottom_left);
            marker_msgs[stixel_msg.semantic_class].points.push_back(top_right);
            marker_msgs[stixel_msg.semantic_class].points.push_back(top_left);

            // Lower right triangle of the stixel's quadrilateral.
            marker_msgs[stixel_msg.semantic_class].points.push_back(bottom_left);
            marker_msgs[stixel_msg.semantic_class].points.push_back(bottom_right);
            marker_msgs[stixel_msg.semantic_class].points.push_back(top_right);
         }
    }
    msg_markerarray.markers = marker_msgs;
}

void InstanceStixelsVisualizationNode::callback(
        const instance_stixels_msgs::InstanceStixelsArrayConstPtr& stixels_msg,
        const ImageConstPtr& image_msg) {
    // Get images from messages.
    cv::Mat image = cv_bridge::toCvCopy(image_msg, "rgb8")->image;
    //cvtColor(image, image, CV_BGR2RGB);

    // Resize/crop both images.
    const auto& roi_msg = stixels_msg->valid_region;
    cv::Rect valid_roi(roi_msg.x_offset, roi_msg.y_offset, 
                       roi_msg.width, roi_msg.height);
    image = image(valid_roi);

    // 3D point cloud of stixel corners
    int n_object_stixels = 0;
    std::vector<unsigned char> byteVec;
    for(auto stixel_msg : stixels_msg->stixels){
        if(add_class_to_pointcloud_[stixel_msg.semantic_class]){
            const auto& vertices = stixel_msg.vertices;
            const unsigned char* bytes =
                reinterpret_cast<const unsigned char*>(&vertices.data()[0]);
            byteVec.insert(
                    byteVec.end(),
                    bytes,
                    bytes + sizeof(float)*vertices.size());
            n_object_stixels++;
        }
    }
    //std::cout << "byteVec.size() = " << byteVec.size() << "\n";

    sensor_msgs::PointCloud2 msg_pointcloud;
    msg_pointcloud.header = stixels_msg->header;
    msg_pointcloud.fields.push_back(x_field_);
    msg_pointcloud.fields.push_back(y_field_);
    msg_pointcloud.fields.push_back(z_field_);
    msg_pointcloud.height = 1;
    msg_pointcloud.width = n_object_stixels * 4;
    msg_pointcloud.data = byteVec;
    msg_pointcloud.is_bigendian = false; // x86-64 is little endian
    msg_pointcloud.point_step = sizeof(float) * 3; // 3 channels
    msg_pointcloud.row_step = msg_pointcloud.width * msg_pointcloud.point_step;
    msg_pointcloud.is_dense = true; // ignore for now that sky is invalid

    // Visualize
    auto stixels_img = DrawStixels(*stixels_msg, image_color_scheme_);
    cv::addWeighted(stixels_img, image_overlay_alpha_,
                    image, 1-image_overlay_alpha_,
                    0.0, stixels_img);
    sensor_msgs::ImagePtr msg_image_stixel =
        cv_bridge::CvImage(image_msg->header, "rgb8", stixels_img).toImageMsg();

    // Create MarkerArray Message
    visualization_msgs::MarkerArray msg_markerarray;
    populateSemanticMarkerArray(msg_markerarray, stixels_msg);

    pub_pointcloud_.publish(msg_pointcloud);
    pub_markerarray_.publish(msg_markerarray);
    pub_image_stixel_.publish(msg_image_stixel);
}

void InstanceStixelsVisualizationNode::reconfigure_callback(
        const instance_stixels::InstanceStixelsVisualizationConfig& config,
        uint32_t level) {
    // TODO: So far I did not put a lock on any of these objects, because the
    // values that are changed are usually just read. Changes within a frame
    // shouldn't have major impact. Change this in the future.
    ROS_INFO("Reconfigure Instance Stixels visualization: %s.",
                config.classes_in_pointcloud.c_str());
    // Classes visible in pointcloud
    for(int i = 0; i < CLASSNAMES_CITYSCAPES.size(); i++) {
        add_class_to_pointcloud_[i] = false;
        if(config.classes_in_pointcloud.find(CLASSNAMES_CITYSCAPES[i])
            != std::string::npos){

            add_class_to_pointcloud_[i] = true;
        }
    }

    // Image color scheme
    image_color_scheme_ = static_cast<ColorScheme>(config.image_color_scheme);

    image_overlay_alpha_ = config.image_overlay_alpha;
}

InstanceStixelsVisualizationNode::InstanceStixelsVisualizationNode(
        ros::NodeHandle nh) 
        :
        nh_(nh), sync_(VisualizationNodeSyncPolicy(10)){
    x_field_.name = "x";
    x_field_.offset = 0;
    x_field_.datatype = sensor_msgs::PointField::FLOAT32;
    x_field_.count = 1;
    y_field_.name = "y";
    y_field_.offset = 4;
    y_field_.datatype = sensor_msgs::PointField::FLOAT32;
    y_field_.count = 1;
    z_field_.name = "z";
    z_field_.offset = 8;
    z_field_.datatype = sensor_msgs::PointField::FLOAT32;
    z_field_.count = 1;
    
    // subscribe
    sub_stixels_.subscribe(nh_, "instance_stixels/stixels", 1);
    sub_image_left_.subscribe(nh_,"left/image_color", 1);

    sync_.connectInput(sub_stixels_, sub_image_left_);
    sync_.registerCallback(
            boost::bind(
                &InstanceStixelsVisualizationNode::callback, this, _1, _2));

    //// publish
    image_transport::ImageTransport it(nh_);
    pub_image_stixel_ = it.advertise("instance_stixels/image", 1);

    pub_pointcloud_ =
        nh_.advertise<sensor_msgs::PointCloud2>(
                "instance_stixels/pointcloud", 100);
    pub_markerarray_ =
        nh_.advertise<visualization_msgs::MarkerArray>(
                "instance_stixels/markers", 100);
    
    // setup dynamic reconfigure server
    dynamic_reconfigure::Server<
        instance_stixels::InstanceStixelsVisualizationConfig>::CallbackType
            f;
    cfg_server_.setCallback(boost::bind(
                &InstanceStixelsVisualizationNode::reconfigure_callback, 
                this, _1, _2));
}



