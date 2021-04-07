// This file is part of Instance Stixels:
// https://github.com/tudelft-iv/instance-stixels
//
// Copyright (c) 2020 Thomas Hehn.
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

#ifndef INSTANCESTIXELS_TYPES_H_
#define INSTANCESTIXELS_TYPES_H_

constexpr int GROUND = 0;
constexpr int OBJECT = 1;
constexpr int SKY    = 2;

// Parameters to configure Stixels class
// Simple way to use a default set of parameters.
// Note: this configuration is used to compute StixelParameters.
// There is overlap between the classes, but they are not the same.
struct StixelConfig {
    // --- Parameters without default values! Need to be set!
    // (Disparity) image properties
    float rows = -1;
    float cols = -1;
    int max_dis = -1;
    float invalid_disparity = -1.0f;
    // invalid_disparity should be castable to interger due to its use as
    // index for Look-Up Tables.
    // < 0 -> no explicit invalid disparities, outliers are still discarded
    // >= 0 (usually == 0) -> accept explicit disparity value

    // DBSCAN clustering parameters
    float eps = -1;
    int min_pts = -1;
    int size_filter = -1;

    // CNN parameters
    int n_semantic_classes = -1;
    int n_offset_channels = -1;

    // Weights
    float prior_weight = -1;
    float segmentation_weight = -1;
    float instance_weight = -1;
    float disparity_weight = -1;

    // This one is not really considered when loading a config in the Stixel
    // class. It needs to be passed to the Compute(...) function. Here it is
    // just used as a convenience storage.
    bool pairwise = false;

    // Stixel width
    int column_step = -1;

    // Camera parameters
    float focal = -1;
    float baseline = -1;
    float camera_center_x = -1;
    float camera_center_y = -1;

    // --- Some useful camera parameter settings.
    // Especially since image have to cropped/resized.
    //  //ORIGINAL
    //  const float focal = 704.7082f;
    //  const float baseline = 0.8f;
    //  const float camera_center_y = 384.0f;

    //CITYSCAPES (defaults, quite random)
    // !!! Note: These parameters are not valid after resizing.
    //const float focal = 2262.52f * size_factor; // 2225.54f;
    //const float baseline = 0.209313f; // 0.222126f;
    //const float camera_center_y = 513.137f * size_factor; // 519.277f;

    //  //UEYE  (resized to 1624x1020 -- downscale factor ~ 0.8388)
    //  const float focal = 1495.46f * size_factor;           //1254.399f;
    //  const float baseline = 0.22087f;
    //  const float camera_center_y = 624.896 * size_factor;     //524.163f;

    //  //KITTI
    //  const float focal = 721.5f;
    //  const float baseline = 0.54f;
    //  const float camera_center_y = 173.0f;

    // --- Parameters with default values. Only change if you care.
    /* Disparity Parameters */
    float sigma_disparity_object = 1.0f;
    float sigma_disparity_ground = 2.0f;
    float sigma_sky = 0.1f; // Should be small compared to sigma_dis

    /* Probabilities */
    // Similar to values in Pfeiffer 14 dissertation, page 49.
    float pout = 0.15f;
    float pout_sky = 0.4f;
    float pord = 0.2f;
    float pgrav = 0.1f;
    float pblg = 0.04f;

    // 0.36, 0.3, 0.34 are similar to values in Pfeiffer 14 dissertation,
    // page 49.
    // However, unequal weighting did lead to invalid regions being classified as
    // ground or sky and instead of continuing an object.
    // Must add to 1.
    float pground_given_nexist = 0.28;//1.f/3.;//0.36f;
    float pobject_given_nexist = 0.44;//1.f/3.;//0.3;
    float psky_given_nexist = 0.28;//1.f/3.;//0.34f;
    // tested: 0.2; 0.6; 0.2; but did not have significant effect.

    // Used this value from Pfeiffer 14 dissertation, page 49.
    float pnexist_dis = 0.25f; // 0.0f;
    float pground = 1.0f/3.0f;
    float pobject = 1.0f/3.0f;
    float psky = 1.0f/3.0f;
    // tested: 0.25; 0.5; 0.25; but did not have significant effect.

    // Virtual parameters
    //const int column_step = 5;
    // Ignore a margin on the left side of the image.
    int width_margin = 0;

    float sigma_camera_tilt = 0.05f;
    float sigma_camera_height = 0.05f;
    //const float camera_center_x = 651.216186523f;

    /* Model Parameters */
    bool median_join = false;
    float epsilon = 3.0f;
    float range_objects_z = 10.20f; // in meters

    /* Road estimation parameter */
    float road_vdisparity_threshold = 0.2f;
};


// Parameters passed to GPU kernels
struct StixelParameters {
    int vhor;
    int rows;
    int rows_power2;
    int rows_power2_segmentation;
    int cols;
    int max_dis;
    float rows_log;
    float pnexists_given_sky_log;
    float normalization_sky;
    float inv_sigma2_sky;
    float puniform_sky;
    float nopnexists_given_sky_log;
    float pnexists_given_ground_log;
    float puniform;
    float nopnexists_given_ground_log;
    float pnexists_given_object_log;
    float nopnexists_given_object_log;
    float baseline;
    float focal;
    float range_objects_z;
    float pord;
    float epsilon;
    float pgrav;
    float pblg;
    float max_dis_log;
    int max_sections;
    int width_margin; // TODO: Remove all occurences unless needed(10-12-2018).
    int segmentation_classes;
    int segmentation_channels;
    float prior_weight;
    float disparity_weight;
    float segmentation_weight;
    float instance_weight;
    int column_step;
    float clustering_eps;
    int clustering_min_pts;
    int clustering_size_filter;
    float invalid_disparity;
};

struct Section {
    int type;
    int vB, vT;
    float disparity;
    int semantic_class;
    float cost;
    float instance_meanx;
    float instance_meany;
};

struct StixelsData {
    std::vector<Section> sections;

    int rows, cols;
    int realcols, max_sections, max_dis;
    int column_step;
    int semantic_classes;
    float alpha_ground;
    int vhor;
};

#endif // INSTANCESTIXELS_TYPES_H_
