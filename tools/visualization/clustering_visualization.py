#!/usr/bin/env python

# This file is part of Instance Stixels.
# Copyright (C) 2019 Thomas Hehn. All right reserved.
#
# Instance Stixels is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Instance Stixels is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import sys
import os
import glob
import json
import copy
import multiprocessing
import itertools

import numpy as np
# TODO: Probably cv2 should be replaced by pillow, which should provide the
# same functionality and is necessary to save 32bit float tiff files. Also, it
# might provide better error messages.
import cv2
import PIL.Image
import h5py as h5

from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

import cityscapesscripts.helpers.labels as cityscapes_labels
from cityscapes_instance_loader import load_instance_mask

#import ipdb # Importing it will result in sequential processing

# --- Color look up tables
trainId2color = [label.color for label in cityscapes_labels.labels\
                 if label.trainId not in (-1,255)]

COLOR_LUT = np.zeros((256,1,3), dtype=np.uint8)
# COLOR_LUT[0] is used for stixel frame -> (0,0,0)
COLOR_LUT[0,0,:] = 255
COLOR_LUT[1:20,0,:] = np.array(trainId2color).astype(np.uint8)
COLOR_LUT_RGB = np.array(COLOR_LUT, copy=True)
COLOR_LUT = cv2.cvtColor(COLOR_LUT, cv2.COLOR_RGB2BGR)

#COLOR_LUT_CV2 = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_HSV)
#COLOR_LUT_CV2 = cv2.cvtColor(COLOR_LUT, cv2.COLOR_RGB2BGR)
COLOR_LUT_RANDOM = np.random.randint(255, size=(217,3))
#~ COLOR_LUT_CV2[0,0,:] = (0,0,0)
#~ # If you want to use a CV2 colormap stretch the label image
#~ semantic_image = semantic_image.astype(np.float) * 255./semantic_image.max()

def encode_color(color, value, max_value):
    #color = cv2.cvtColor(color, cv2.COLOR_RGB2YCrCb)
    #color = cv2.cvtColor(color, cv2.COLOR_YCrCb2RGB)
    color = np.array([[color]], dtype=np.uint8)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2HLS)
    # We're loosing some precision here due uint8 rounding, but that's fine.
    color[0,0,1] = min(255, ((value / max_value) * 0.8 + 0.3) * 255)
    color = cv2.cvtColor(color, cv2.COLOR_HLS2RGB)
    return color.squeeze()

def read_stixel_file(filename):
    stixels = []
    groundplane = None
    with open(filename, 'r') as stixelfile:
        # each line represents a column
        for line in stixelfile:
            if line.startswith('groundplane'):
                groundplane = line[11:].split(",")
                groundplane = (float(groundplane[0]), int(groundplane[1]))
                continue
            stixels.append([]) # add column
            stixels_column = line.split(';')
            for stixel in stixels_column:
                if stixel == '\n': # skip newline
                    break
                stixel = stixel.split(',')

                # no semantic information
                stixel_entry = {
                    'type'      : int(stixel[0]),
                    'vB'        : int(stixel[1]),
                    'vT'        : int(stixel[2]),
                    'disparity' : float(stixel[3])}
                if len(stixel) >= 5: # with semantic information
                    stixel_entry['class'] = int(stixel[4])
                if len(stixel) >= 6: # with semantic and cost information
                    stixel_entry['cost'] = float(stixel[5])
                if len(stixel) >= 8: # with semantic, cost and instance mean
                    stixel_entry['instance_mean_x'] = float(stixel[6])
                    stixel_entry['instance_mean_y'] = float(stixel[7])
                if len(stixel) >= 9:
                    if stixel_entry['class'] < 11:
                        raise("Error: class {} should not have instance "
                              "label".
                              format(stixel_entry['class']))
                    # convert to cityscapes style
                    instance_label = int(stixel[8])
                    if instance_label >= 0 and instance_label < 1000:
                        instance_label += stixel_entry['class']*1000
                    else:
                        instance_label = -1
                    stixel_entry['instance_label'] = instance_label
                stixels[-1].append(stixel_entry)
    return stixels, groundplane

def draw_instance_masks(stixels, mask_shape):
    stixel_width = mask_shape[1] // len(stixels)
    masks = {}
    for col, column_stixels in enumerate(stixels):
        for stixel in column_stixels:
            if ('instance_label' in stixel.keys()
                and stixel['instance_label'] > 0):
                # Ignore non-instance stixels.
                instance_id = stixel['instance_label']
                if instance_id not in masks.keys():
                    masks[instance_id] = np.zeros(mask_shape, dtype=np.uint8)
                instance_mask = masks[instance_id]

                topleft_x = col * stixel_width
                topleft_y = mask_shape[0] - stixel['vT'] - 1 # vT, mirror y-axis
                bottomright_x = topleft_x + stixel_width - 1 # e.g. width 5: 0-4,5-9,
                bottomright_y = mask_shape[0] - stixel['vB'] - 1 # vB, mirror y-axis

                cv2.rectangle(instance_mask,
                              (topleft_x, topleft_y),
                              (bottomright_x, bottomright_y),
                              255,
                              thickness=-1) # fill rectangle

    return masks

def save_instance_masks(instance_mask_dir, filename_base, masks):
    txtlines = []
    for id_, mask in masks.items():
        mask_file = "{}_{}.png".format(filename_base, id_)
        cv2.imwrite(os.path.join(instance_mask_dir, mask_file), mask)

        class_trainid = id_ // 1000
        class_id = cityscapes_labels.trainId2label[class_trainid].id
        if class_id < 24:
            raise ValueError("These classes should not have instances. "
                             "Class id = {}, class train id = {}."
                             .format(class_id, class_trainid))
        txtlines.append("{} {} 1.0".format(mask_file, class_id))

    with open("{}/{}.txt".format(instance_mask_dir, filename_base), 'w') as f:
        for line in txtlines:
            f.write(line+'\n')

# TODO: clean this up... it has too many return values
# TODO: Refactor: disparity image is actually distance image...
def draw_stixels(stixels, max_disparity,
        disparity_image=None, semantic_image=None, semantic_labelImg=None,
        cost_image=None, max_cost=1e4, instancegt_mask=None,
        disparity_result_image=None, instance_image=None,
        median_disparity_image=None):
    """
    This method visualizes the stixels as images. Output images are for example
    a semantic segmentation image which visualizes each class by a color.
    A stixel will then be drawn as a rectangle. Optionally, a border can be
    drawn around the stixel. The final result can then be overlayed with the
    original RGB image.

    Note: Over the course of implementing and especially extending the
    proof of conecpt, it has become quite a mess, but it works.
    """
    instance_gt_image = None
    shape = None
    for img in (semantic_image, disparity_image, semantic_labelImg):
        if img is not None:
            shape = img.shape[:2]
    if shape is None:
        raise ValueError("At least one output image has to be provided!")
    stixel_width = shape[1] // len(stixels)
    print("Stixel width = {}".format(stixel_width))
    print("image shape = {}".format(shape))

    stixel_count = 0
    for col, column_stixels in enumerate(stixels):
        for stixel in column_stixels:
            stixel_count += 1
            # Decide color based upon stixel type.
            color = (255,255,255) # type 0 (ground)
            instance_color = np.array((0,0,0,255),dtype=np.float)
            #cost = stixel[5] / max
            min_cost = -1e4
            max_cost = 1e4
            factor = max((stixel['cost'] - min_cost),0) / (max_cost - min_cost)
            cost = (1. - factor) * 255

            # Compute image coordinates
            # *.stixels file    OpenCV
            # row in [0,h-1]    y in [0,h-1]
            #  ^                0/0---x---->
            #  |                 |
            #  |                 |
            # row                y
            #  |                 |
            #  |                 |
            # 0/0--column-->     v
            # y = (h-1) - row
            topleft_x = col * stixel_width
            topleft_y = shape[0] - stixel['vT'] - 1 # vT, mirror y-axis
            bottomright_x = topleft_x + stixel_width - 1 # e.g. width 5: 0-4,5-9,
            bottomright_y = shape[0] - stixel['vB'] - 1 # vB, mirror y-axis
            #if stixel[1] == stixel[2]:
            #    print("Size 1!")

            # type distinction
            if stixel['type'] == 2:
                # Type 0 (sky).
                color = (255,191,0)
            elif stixel['type'] == 1:
                # Type 1 (object).
                # Cropping of at disparities of max_disparity * 2e-2 = 2.56 and
                # below.
                # TODO: Is this now linear?
                #distance = min((max_disparity/float(stixel[3]) * 4.5e-2),1.0)
                distance = min(float(stixel['disparity']+20)/max_disparity, 1.)
                #color = disparity*np.array([[[255.,255.,255.]]])
                #color = np.array([[cm.autumn(1.-distance)[:3]]])*255
                color = np.array([[cm.inferno(distance)[:3]]])*255
                #print(np.max(distance), np.min(distance))
                #np.sort(
                #color = np.array([[0.5 *\
                #                   (np.array([0.,1.,0.]) * distance +\
                #                    np.array([1.,0.,0.]) * (1.-distance))]])*255
                #color = np.array([[0.5 *\
                #                   (np.array([0.,255.,90.]) * distance +\
                #                    np.array([0.,0.,195.]) * (1.-distance))]])
                                   #(np.array([0.,255.,255.]) * distance +\
                                   # np.array([60.,255.,255.]) * (1.-distance))]])
                color = color.astype(np.uint8)
                #color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                color = color[0,0].tolist()
                #color = np.uint8([[[disparity*0.3*255, 255, 255]]])
                #color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
                #print("color = {}".format(color))
                #color = color[0,0].tolist()

                # compute instance affiliation from ground truth
                if instancegt_mask is not None:
                    # Only consider classes with instances.
                    if stixel['class'] < 11: # 10 is sky
                        instance_color = np.array((0.,0.,0.,255.))
                    else:
                        max_instanceid = 0
                        # Must be at least 10% of stixel.
                        max_instanceid_pixels = 0
                        # Only use instance mask of predicted class.
                        # Before:for mask in instancegt_mask:
                        mask = instancegt_mask[stixel['class']-11]
                        # extract pixels that belong to stixel from mask
                        stixel_pixels = mask[topleft_y:bottomright_y+1,
                                             topleft_x:bottomright_x+1]\
                                            .astype(np.int)
                        pixels_per_id = np.bincount(np.ravel(stixel_pixels))

                        # ignore zero-rows-stixels
                        if len(pixels_per_id) > 0:
                            max_instanceid_pixels = pixels_per_id.max()
                            max_instanceid = pixels_per_id.argmax()

                        instance_color =\
                                COLOR_LUT_RANDOM[max_instanceid\
                                                 % len(COLOR_LUT_RANDOM),
                                                 :]
                        # Alpha based on number of instance pixels.
                        #alpha = np.clip(max_instanceid_pixels, 150.0, 255.0)
                        alpha = 255.0
                        instance_color = np.concatenate([instance_color,
                                                         [alpha]])
                        instance_color = instance_color.astype(np.float)
                        #instance_color *=\
                        #    np.minimum(max_instanceid_pixels/(stixel_width*50.), 1.0)
                        if max_instanceid == 0:
                            instance_color[:3] = 255.0
                            instance_color[3] = 255.0
                        elif (max_instanceid_pixels <
                                0.1*stixel_width*(bottomright_y-topleft_y)):
                            instance_color[:3] = 255.0
                        #instance_colors[ofinstances == 0,:] = 0
                        #marker = MARKERS[max_instanceid % len(MARKERS)]

            if instancegt_mask is not None:
                if instance_gt_image is None:
                    instance_gt_image = np.zeros((*shape,4))
                cv2.rectangle(instance_gt_image,
                              (topleft_x, topleft_y),
                              (bottomright_x, bottomright_y),
                              instance_color,
                              thickness=-1) # fill rectangle
                # let's skip borders for now
                if False:
                    if np.abs(topleft_y - bottomright_y) > 1:
                        cv2.rectangle(instance_gt_image,
                                      (topleft_x, topleft_y),
                                      (bottomright_x+1, bottomright_y+1),
                                      0,
                                      thickness=1) # border only on left side
            if instance_image is not None:
                if 'instance_label' in stixel.keys():
                    instance_id = int(stixel['instance_label'])
                    if instance_id == -1:
                        instance_color = np.array((255., 255., 255.))
                    else:
                        instance_color = COLOR_LUT_RANDOM[instance_id
                                                           % len(COLOR_LUT_RANDOM),
                                                          :]
                    instance_color = instance_color.astype(np.float)
                    cv2.rectangle(instance_image,
                                  (topleft_x, topleft_y),
                                  (bottomright_x, bottomright_y),
                                  instance_color,
                                  thickness=-1) # fill rectangle
                if True: # full border
                    if np.abs(topleft_y - bottomright_y) > 1:
                        cv2.rectangle(instance_image,
                                      (topleft_x, topleft_y),
                                      (bottomright_x+1, bottomright_y+1),
                                      (255.0,255.0,255.0),
                                      thickness=1)
                if False: # upper end border
                    if np.abs(topleft_y - bottomright_y) > 1:
                        cv2.rectangle(instance_image,
                                      (topleft_x, topleft_y),
                                      (bottomright_x, topleft_y-2),
                                      (255,255,255),
                                      thickness=-1)
            if median_disparity_image is not None:
                cv2.rectangle(median_disparity_image,
                              (topleft_x, topleft_y),
                              (bottomright_x, bottomright_y),
                              255-int(stixel['instance_disparity']),
                              thickness=-1)
            if disparity_image is not None:
                cv2.rectangle(disparity_image,
                              (topleft_x, topleft_y),
                              (bottomright_x, bottomright_y),
                              color,
                              thickness=-1) # fill rectangle
                # Border always overlaps with right and bottom neighboring
                # stixel.
                #if np.abs(topleft_y - bottomright_y) > 2:
                #    cv2.rectangle(disparity_image,
                #                  (topleft_x, topleft_y),
                #                  (bottomright_x+1, bottomright_y+1),
                #                  (0,0,0),
                #                  thickness=1) # border only on left side
            if semantic_image is not None and 'class' in stixel.keys():
                class_color = COLOR_LUT[stixel['class']+1].squeeze()

                use_distance_colors = True
                if use_distance_colors and stixel['class'] >= 11:
                    class_color = encode_color(class_color, stixel['disparity'], 128)
                    class_color = class_color.astype(float).squeeze()
                cv2.rectangle(semantic_image,
                              (topleft_x, topleft_y),
                              (bottomright_x, bottomright_y),
                              class_color.tolist(),
                              thickness=-1) # fill rectangle
                if True: # one pixel width border (around)
                    if np.abs(topleft_y - bottomright_y) > 1:
                        cv2.rectangle(semantic_image,
                                      (topleft_x, topleft_y),
                                      (bottomright_x+1, bottomright_y+1),
                                      (255,255,255),
                                      thickness=1) # border only on left side
                if False: # upper end border of "width" pixels
                    width = 0
                    if np.abs(topleft_y - bottomright_y) > 1:
                        cv2.rectangle(semantic_image,
                                      (topleft_x, topleft_y),
                                      (bottomright_x, topleft_y-width),
                                      (255,255,255),
                                      thickness=-1) # border only on left side
            if cost_image is not None and 'cost' in stixel.keys():
                cv2.rectangle(cost_image,
                              (topleft_x, topleft_y),
                              (bottomright_x, bottomright_y),
                              int(cost),
                              thickness=-1) # fill rectangle
            # --- result images
            if semantic_labelImg is not None and 'class' in stixel.keys():
                cv2.rectangle(semantic_labelImg,
                              (topleft_x, topleft_y),
                              (bottomright_x, bottomright_y),
                              cityscapes_labels.trainId2label[stixel['class']].id,
                              thickness=-1) # fill rectangle
            if disparity_result_image is not None:
                # TODO: Handle ground stixels appropriately.
                cv2.rectangle(disparity_result_image,
                              (topleft_x, topleft_y),
                              (bottomright_x, bottomright_y),
                              stixel['disparity'],
                              thickness=-1) # fill rectangle
    print("Processed {} stixels.".format(stixel_count))
    return (disparity_image, semantic_image, semantic_labelImg, cost_image,
            stixel_count, instance_gt_image, disparity_result_image,
            instance_image, median_disparity_image)

def pointcloud(stixels, image_shape, max_disparity, groundplane,
               camera_parameters, instancegt_mask=None):
    stixel_width = image_shape[1] // len(stixels)
    stixel_count = 0
    points = []
    pixels = []
    osemantics = []
    gsemantics = []
    ground_patches = []
    instances = []
    for col, column_stixels in enumerate(stixels):
        for stixel in column_stixels:
            stixel_count += 1
            # Compute coordinates
            # *.stixels file    OpenCV
            # row in [0,h-1]    y in [0,h-1]
            #  ^                0/0---x---->
            #  |                 |
            #  |                 |
            # row                y
            #  |                 |
            #  |                 |
            # 0/0--column-->     v
            # y = (h-1) - row
            topleft_x = col * stixel_width
            topleft_y = image_shape[0] - stixel['vT'] - 1 # vT, mirror y-axis
            bottomright_x = topleft_x + stixel_width - 1 # e.g. width 5: 0-4,5-9,
            bottomright_y = image_shape[0] - stixel['vB'] - 1 # vB, mirror y-axis

            mean_x = 0.5 * (topleft_x + bottomright_x + 1) # e.g. stixel 0-5: u=2.5
            mean_y = (0.5 * (topleft_y + bottomright_y))

            # Decide color based upon stixel type.
            if stixel['type'] == 0: # type 0 (ground)
                # Expects row in [0,h-1] starting from bottom. See
                # GroundFunction in Stixels.cu.
                ground_disparity =\
                    lambda row: groundplane[0] * float(groundplane[1]-row)
                top_disparity = ground_disparity(stixel['vT'])
                bottom_disparity = ground_disparity(stixel['vB'])
                ground_patches.append(
                        [[topleft_x, topleft_y, top_disparity],
                         [bottomright_x, topleft_y, top_disparity],
                         [bottomright_x, bottomright_y, bottom_disparity],
                         [topleft_x,   bottomright_y, bottom_disparity]])
                gsemantics.append(stixel['class'])
            # type 1 (object) and disparity > 1.0 (otherwise sky)
            elif stixel['type'] == 1:
                # append to corresponding lists
                pixels.append(bottomright_y - topleft_y + 1) # number of rows
                points.append([mean_x,mean_y,stixel['disparity']]) # stixel[3] = mean disparity
                osemantics.append(stixel['class'])

                if 'instance_label' in stixel.keys():
                    instances.append(stixel['instance_label'])
                elif (stixel['class'] > 10 and
                      'instance_disparity' in stixel.keys() and
                      stixel['instance_disparity'] == 0):
                    instances.append(-1)
                #~ using ground truth
                #~~# compute instance affiliation from ground truth
                #~~if instancegt_mask is not None:
                #~~    max_instanceid = 0
                #~~    max_instanceid_pixels = 0
                #~~    # TODO: iterate only over stixel[4] semantic class to condition
                #~~    # on class prediction
                #~~    for mask in instancegt_mask: # separate classes
                #~~        # Extract pixels that belong to stixel from mask.
                #~~        # +1 to include the border pixels e.g.: [0,4] instead
                #~~        # of [0,4).
                #~~        stixel_pixels = mask[topleft_y:bottomright_y+1,
                #~~                             topleft_x:bottomright_x+1]\
                #~~                            .astype(np.int)
                #~~        pixels_per_id =\
                #~~                np.bincount(np.ravel(stixel_pixels))

                #~~        # ignore zero-rows-stixels
                #~~        if len(pixels_per_id) > 0:
                #~~            # ignore 0 labels
                #~~            pixels_per_id[0] = 0
                #~~            # check if max instance in stixel region is new max
                #~~            # instance
                #~~            if pixels_per_id.max() > max_instanceid_pixels:
                #~~                max_instanceid_pixels = pixels_per_id.max()
                #~~                max_instanceid = pixels_per_id.argmax()
                #~~    instances.append(max_instanceid)

    # wrap in numpy array
    points = np.array(points)
    pixels = np.array(pixels)
    osemantics = np.array(osemantics)
    gsemantics = np.array(gsemantics)
    instances = np.array(instances)
    ground_patches = np.array(ground_patches)

    points3d = compute3d(points, camera_parameters)
    ground_patches3d = compute3d(ground_patches.reshape(-1,3),
                                 camera_parameters)
    ground_patches3d = ground_patches3d.reshape(*ground_patches.shape[:2], 3)

    return {'points3d' : points3d, 'points' : points,
            'pixels' : pixels, 'ground_patches3d': ground_patches3d,
            'ground_semantics' : gsemantics, 'object_semantics' : osemantics,
            'instances' : instances}

# TODO: needs checking!
def compute3d(points, camera_parameters):
    """
    points : 2D image (OpenCV) coordinates as shown below including a disparity
        measurement (0,128] at index 2.
        OpenCV coordinates:
        y in [0,h-1]
        0/0---x---->
         |
         |
         y
         |
         |
         v
        u0, v0 are in the same coordinate frame according to
        https://github.com/mcordts/cityscapesScripts/blob/master/docs/csCalibration.pdf
    """
    if camera_parameters is None:
        print("\n\nScale factor of camera parameters is not included!!!!\n\n")
        camera_parameters = {'fx' : 0.766* 2262.52, # cm?
                             'fy' : 0.766* 2265.30, # cm?
                             'u0' : 0.766* 1096.98, # px
                             'v0' : 0.766* 513.137, # px
                             'baseline' : 0.209313} # m
        raise ValueError("Camera parameters can not be None anymore!")

    points3d = np.empty(points.shape)
    if np.any(points[:,2] == 0):
        # TODO: Handle these cases approriately.
        raise ValueError("Divide by zero. Disparity should not be 0.")
    # distance to camera -> z coordinate
    points3d[:,2] = (camera_parameters['intrinsic']['fx']
                     * camera_parameters['extrinsic']['baseline'])\
                    / points[:,2]
    # horizontal positon w.r.t to the camera -> x coordinate
    points3d[:,0] = -(points3d[:,2] / camera_parameters['intrinsic']['fx'])\
                    * (camera_parameters['intrinsic']['u0'] - points[:,0])
    # vertical positon w.r.t to the camera -> y coordinate
    points3d[:,1] = -(points3d[:,2] / camera_parameters['intrinsic']['fy']) *\
                       (camera_parameters['intrinsic']['v0'] - points[:,1])
    return points3d

# TODO: needs checking!
def plot_topdownview(pc, filter_mask, size_filter=0):
    # draw image as stixels
    # opoints = points[~ground_mask]
    # gpoints = points[ground_mask]
    # plt.errorbar(gpoints[:,0],gpoints[:,1],yerr=0.5*pixels[ground_mask],xerr=None,linestyle=None,lw=0,marker='o',elinewidth=4.0,c='r')
    # plt.errorbar(opoints[:,0],opoints[:,1],yerr=0.5*pixels[~ground_mask],xerr=None,linestyle=None,lw=0,marker='o',elinewidth=4.0,c='b')

    # onpoints = points3d[~ground_mask]
    # gnpoints = points3d[ground_mask]
    # plt.scatter(onpoints[:,0],onpoints[:,2],s=0.1*pixels[~ground_mask],c='b')
    # plt.scatter(gnpoints[:,0],gnpoints[:,2],s=0.1*pixels[ground_mask],c='r')

    # we're happy with 200 rows
    scaling = lambda x: 4.0*np.maximum(np.minimum(x/200.,1),
                                       0.3)
    points3d = pc['points3d']
    points = pc['points']
    pixels = pc['pixels']

    # TODO: refactor... rename all "n" (new?) points to 3d? e.g. onpoints,
    # onfpoints, ...
    opoints = points
    onpoints = points3d
    opixels = pixels
    osemantics = pc['object_semantics']
    gsemantics = pc['ground_semantics']
    oinstance_ids = pc['instances']
    oinstance_ids = oinstance_ids if len(oinstance_ids) > 0 else None

    filter_mask = filter_mask(onpoints)
    # Filter MB-Star: for 960 < x < 1160 and the lowest 100 pixels
    filter_mask[((opoints[:,0] > 960) & (opoints[:,0] < 1160)) & (opoints[:,1] > 684)] = False

    ofpoints = opoints[filter_mask]
    onfpoints = onpoints[filter_mask]
    ofpixels = opixels[filter_mask]
    ofsemantics = osemantics[filter_mask]
    if oinstance_ids is not None:
        ofinstance_ids = oinstance_ids[filter_mask[osemantics > 10]]
        instance_idxs = np.where(ofsemantics > 10)[0]
        non_instance_idxs = np.where(ofsemantics <= 10)[0]

    #colors_list = [(semantic_colors,['o']*len(semantic_colors)),
    #               (instance_colors,instance_markers)] # semantics + 8 instance classes

    #colors = np.concatenate((colors, np.ones((colors.shape[0],1)) ), axis=1)
    #plt.errorbar(ofpoints[:,0],ofpoints[:,1],yerr=0.5*ofpixels,xerr=None,
    #             linestyle=None,lw=0,marker='x',elinewidth=4,ecolor=colors,
    #             markeredgecolor='white',markersize=2.0)
    #plt.figure()

    figs = []
    figsize = (4,12)
    max_distance = 50

    # --- first plot semantics
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    ## plot ground stixel polygons
    plot_ground = True
    ground_alpha = 0.2
    if plot_ground:
        for (polygon_points, label) in zip(pc['ground_patches3d'], gsemantics):
            color = np.array(trainId2color[label]) / 255.
            patch = plt.Polygon(polygon_points[:,[0,2]], zorder=1, color=color, alpha=ground_alpha)
            ax.add_patch(patch)

    # get class specific colors
    semantic_colors = map(lambda s: trainId2color[s], ofsemantics)
    semantic_colors = np.array(list(semantic_colors)).astype(float) / 255.

    # plot 2D object stixel points
    ax.scatter(onfpoints[:,0],onfpoints[:,2],
               s=scaling(ofpixels),
               c=semantic_colors,zorder=2,edgecolors='none')
    figs.append((fig,ax))

    for i in range(2):
        if oinstance_ids is not None:
            # --- then plot instances
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize) #, figsize=[4,8])

            if i == 0:
                ax.scatter(onfpoints[non_instance_idxs,0],onfpoints[non_instance_idxs,2],
                           s=scaling(ofpixels[non_instance_idxs]),
                           c=semantic_colors[non_instance_idxs],zorder=2,edgecolors='none')
            if plot_ground:
                ## plot ground stixel polygons
                for (polygon_points, label) in zip(pc['ground_patches3d'], gsemantics):
                    color = np.array(trainId2color[label]) / 255.
                    patch = plt.Polygon(polygon_points[:,[0,2]], zorder=1, color=color, alpha=ground_alpha)
                    ax.add_patch(patch)

            # set up random, different colors and markers per instance
            #instance_colors = [COLOR_LUT_RANDOM[s % len(COLOR_LUT_RANDOM),:]\
            #                    for s in ofinstance_ids]
            #instance_colors = np.array(list(instance_colors)).astype(float) / 255.
            #instance_colors[ofinstance_ids == 0,:] = 0
            MARKERS = ['o','x','v','+','s','^','>']
            #instance_marker_idxs = ofinstance_ids % len(MARKERS)

            # Plot 2D object stixel points.
            # for loop is required since marker can not be a list.
            for instance_id in np.unique(ofinstance_ids):
                #instance_idxs = np.where(instance_marker_idxs == marker_idx)[0]
                instance_id_idxs = instance_idxs[ofinstance_ids == instance_id]
                ofpoints_instance = ofpoints[instance_id_idxs]
                onfpoints_instance = onfpoints[instance_id_idxs]
                ofpixels_instance = ofpixels[instance_id_idxs]
                #colors_instance = instance_colors[instance_idxs]

                # TODO: why is "0" excluded here?
                if instance_id == 0 or instance_id == -1:
                    if i == 0: # plot points
                        ax.scatter(onfpoints_instance[:,0], onfpoints_instance[:,2],
                                   s=scaling(ofpixels_instance),
                                   c=semantic_colors[instance_id_idxs],zorder=2,edgecolors='none')
                    continue


                #ax.scatter(onfpoints_instance[:,0],onfpoints_instance[:,2],
                #           s=3*scaling(ofpixels),
                #           c=color, marker=marker, zorder=2, edgecolors='none')

                plot_stixels_3Dpos = []
                #plot_stixels_disparities = []
                # Connect lowest stixel per instance.
                # Iterate over columns where instance is present.
                for column in np.unique(ofpoints_instance[:,0]):
                    # Get all stixels of this column.
                    column_idxs = np.where(ofpoints_instance[:,0] == column)[0]
                    ofpoints_column = ofpoints_instance[column_idxs]
                    onfpoints_column = onfpoints_instance[column_idxs]
                    ofpixels_column = ofpixels_instance[column_idxs]

                    use_closest_stixel = True
                    if use_closest_stixel:
                        distances = np.sqrt((onfpoints_column**2).sum(axis=1))
                        stixel_idx = np.argmin(distances)
                    else:
                        # Find bottom stixel.
                        # Point should contain a 'y' axis with origin in top left of
                        # the image and thus we need a argmax. It also looked way
                        # better like this. Might be wrong though.
                        #bottom_stixel_idx = np.argmax(ofpoints_column[:,1])
                        # TODO: it's now the largest stixel
                        stixel_idx = np.argmax(ofpixels_column)
                        if ofpixels_column[stixel_idx] < size_filter:
                            continue
                    plot_stixels_3Dpos.append(onfpoints_column[stixel_idx])
                    #plot_stixels_disparities.append(ofpoints_column[stixel_idx,2])

                use_mean_diff = True
                use_laplace = False
                if len(plot_stixels_3Dpos) > 0 and any((use_mean_diff, use_laplace)):
                    plot_stixels_3Dpos = np.atleast_2d(plot_stixels_3Dpos)
                    if use_laplace:
                        laplace = np.zeros_like(plot_stixels_3Dpos)
                        for i, column in enumerate(plot_stixels_3Dpos.T):
                            # repeat border
                            column = np.concatenate(([column[0]], column, [column[-1]]))
                            # TODO:zero 2nd derivate at border?
                            laplace[:,i] = np.convolve([1,-2,1], column, mode='valid')
                        laplace = np.sqrt(np.sum(laplace**2, axis=1))
                        inlier_mask = laplace < 2.0
                    if use_mean_diff:
                        residuals = np.sqrt(np.sum((plot_stixels_3Dpos - plot_stixels_3Dpos.mean(axis=0))**2,axis=1))
                        #inlier_mask = residuals < np.std(residuals) * 3
                        inlier_mask = residuals < 3.0
                    n_old = plot_stixels_3Dpos.shape[0]
                    plot_stixels_3Dpos = plot_stixels_3Dpos[inlier_mask]
                    print("DEBUG: Filtered {} points!".format(n_old-plot_stixels_3Dpos.shape[0]))

                if len(plot_stixels_3Dpos) > 0:
                    plot_stixels_3Dpos = np.atleast_2d(plot_stixels_3Dpos)
                    #mean_distance = plot_stixels_3Dpos[:,2].mean()
                    mean_disparity = np.mean(ofpoints_instance[:,2])

                    use_class_colors = False
                    use_distance_colors = True
                    color = COLOR_LUT_RANDOM[instance_id % len(COLOR_LUT_RANDOM),:]
                    color = np.array(color).astype(float) / 255.
                    if use_class_colors:
                        color = trainId2color[instance_id // 1000]
                        color = np.array(color).astype(float) / 255.
                    if use_distance_colors:
                        # class color:
                        color = trainId2color[instance_id // 1000]
                        #color = encode_color(color, mean_distance, max_distance)
                        color = encode_color(color, mean_disparity, 128)
                        color = color.astype(float).squeeze() / 255.
                        #ax.scatter(onfpoints_instance[:,0],onfpoints_instance[:,2],
                        #           s=scaling(ofpixels_instance),
                        #           c=[color],zorder=2,edgecolors='none')
                    #instance_colors[ofinstance_ids == 0,:] = 0
                    marker = MARKERS[instance_id % len(MARKERS)]

                    #color =
                    ax.plot(plot_stixels_3Dpos[:,0],
                            plot_stixels_3Dpos[:,2],
                            c=color,lw=1.5,alpha=0.9,zorder=99)
                    #ax.plot(plot_stixels_3Dpos[:,0],
                    #        plot_stixels_3Dpos[:,2],
                    #        'o',c=color,ms=1,zorder=4)

            figs.append((fig,ax))
            #for (onfpoint, ofpixel, color, marker) in\
            #        zip(onfpoints, ofpixels, instance_colors, instance_markers):
            #    ax.scatter(onfpoints[:,0],onfpoints[:,2],
            #               s=0.2*np.minimum(ofpixels,20), c=color, marker=marker,
            #               zorder=2, edgecolors='none')
            #ax.scatter(onfpoints[:,0],onfpoints[:,2],s=0.4*np.minimum(ofpixels,40),
            #           c=instance_colors,zorder=2,edgecolors='none')
    for fig, ax in figs:
        ax.grid(True, color='lightgray')
        ax.axis('square')
        #ax.set_xlim([-5,5])
        #ax.set_ylim([10,17])
        ax.set_xlim([-20,20])
        ax.set_ylim([0,max_distance])
        ax.set_ylabel("Longitudial position [m]")
        ax.set_xlabel("Lateral position [m]")
        ax.set_facecolor((1,1,1))#'black')
        #ax.set_facecolor((0.1,0.1,0.1))#'black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

    #plt.show()
    return [fig for fig,_ in figs]

def get_disparity_instance_centers(stixels, class_=None):
    x_instance_pos = []
    y_instance_pos = []
    instance_disparity = []
    size_instance = []
    instance_stixels = []
    for col, column_stixels in enumerate(stixels):
        for stixel in column_stixels:
            if (stixel['class'] > 10 and
                (class_ is None or stixel['class'] == class_)
                and stixel['instance_disparity'] != 0):
                # Only consider stixel with instance class.
                # Already have an instance label from stixel file.
                if 'instance_label' in stixel.keys():
                    instance_stixels.append(stixel)
                    continue
                x_instance_pos.append(stixel['instance_mean_x'])
                y_instance_pos.append(stixel['instance_mean_y'])
                instance_disparity.append(stixel['instance_disparity'])
                size_instance.append(stixel['vT'] - stixel['vB'] + 1)
                # TODO: Refactor. It is kind of nasty to use these pointers to
                # mutables to change stuff in actual stixels list.
                instance_stixels.append(stixel)
    size = np.array(size_instance)
    X = np.array([x_instance_pos, y_instance_pos, instance_disparity])
    return X, size, instance_stixels

def get_instance_means(stixels, class_=None):
    x_instance_pos = []
    y_instance_pos = []
    size_instance = []
    instance_stixels = []
    car_stixels = 0
    for col, column_stixels in enumerate(stixels):
        for stixel in column_stixels:
            if (stixel['class'] > 10 and\
                (class_ is None or stixel['class'] == class_)):
                # Only consider stixel with instance class.
                # Already have an instance label from stixel file.
                if 'instance_label' in stixel.keys():
                    instance_stixels.append(stixel)
                    continue # basically return
                x_instance_pos.append(stixel['instance_mean_x'])
                y_instance_pos.append(stixel['instance_mean_y'])
                size_instance.append(stixel['vT'] - stixel['vB'] + 1)
                # TODO: Refactor. It is kind of nasty to use these pointers to
                # mutables to change stuff in actual stixels list.
                instance_stixels.append(stixel)
    size = np.array(size_instance)
    X = np.array([x_instance_pos, y_instance_pos])
    return X, size, instance_stixels

def assign_instances_gt(stixels, instancegt_mask):
    augmented_stixels = copy.deepcopy(stixels)
    mask_shape = instancegt_mask[0].shape
    stixel_width = mask_shape[1] // len(stixels)

    for col, column_stixels in enumerate(stixels):
        for s_idx, stixel in enumerate(column_stixels):
            if stixel['class'] < 11:
                continue

            max_instanceid = -1
            max_instanceid_pixels = 0
            # Only use instance mask of predicted class.
            # Before: for mask in instancegt_mask:
            mask = instancegt_mask[stixel['class']-11]

            topleft_x = col * stixel_width
            topleft_y = mask_shape[0] - stixel['vT'] - 1 # vT, mirror y-axis
            bottomright_x = topleft_x + stixel_width - 1 # e.g. width 5: 0-4,5-9,
            bottomright_y = mask_shape[0] - stixel['vB'] - 1 # vB, mirror y-axis

            # extract pixels that belong to stixel from mask
            stixel_pixels = mask[topleft_y:bottomright_y+1,
                                 topleft_x:bottomright_x+1]\
                                .astype(np.int)
            pixels_per_id = np.bincount(np.ravel(stixel_pixels))

            # ignore zero-rows-stixels
            if len(pixels_per_id) > 0:
                max_instanceid_pixels = pixels_per_id.max()
                max_instanceid = pixels_per_id.argmax()

            # Note if you change this, you should also changed it in
            # draw_stixels.
            # Must be at least 10% of stixel.
            if (max_instanceid_pixels <
                    0.1*stixel_width*(bottomright_y-topleft_y)):
                max_instanceid = -1
            # id -1 will stay -1, otherwise convert to trainids.
            if max_instanceid > 1000:
                max_instanceid = max_instanceid % 1000 + stixel['class'] * 1000
            else:
                max_instanceid = -1
            augmented_stixels[col][s_idx]['instance_label'] = max_instanceid

    return augmented_stixels

# TODO: Refactor. Very inefficient.
def assign_instances(stixels, cluster_config, camera_parameters=None,
        collect_centers=False):
    eps = cluster_config['eps']
    min_samples = cluster_config['min_size']
    size_filter = cluster_config['size_filter']

    collected_centers = []
    augmented_stixels = copy.deepcopy(stixels)
    for class_ in range(11,19):
        if len(cluster_config['use_instance_disparity']) > 0:
            X, size, instance_stixels =\
                    get_disparity_instance_centers(augmented_stixels, class_)
            xyz_clustering = False
            if xyz_clustering:
                X = compute3d(X.transpose(), camera_parameters).transpose()
        else:
            X, size, instance_stixels =\
                    get_instance_means(augmented_stixels, class_)
        if X.shape[1] == 0:
            if len(instance_stixels) > 0:
                print("INFO: Instance ids already set!")
            else:
                print("INFO: No instance information.")
            continue
        print("INFO: Clustering for class {}".format(class_))
        # Filter small stixels.
        large_mask = size >= size_filter
        large_indices = np.nonzero(large_mask)[0]
        X_large = X[:, large_mask]
        #print("X_large.shape",X_large.shape)

        small_mask = size < size_filter
        small_indices = np.nonzero(small_mask)[0]
        X_small = X[:, small_mask]

        labels = -np.ones(X.shape[1])
        # Given we have a couple of large stixels of that class.
        if X_large.shape[1] > min_samples:
            # Cluster.
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
            dbscan = dbscan.fit(X_large.transpose())
            large_labels = np.array(dbscan.labels_)
            unique_labels = np.unique(large_labels)
            core_indices = np.array(dbscan.core_sample_indices_)

            if core_indices.shape[0] > 0:
                # 2 x core
                X_core = X_large[:, core_indices]
                # 2 x small x core -> small x core
                distance_to_core = ((X_small[:,:,None] - X_core[:,None,:])**2)\
                                   .sum(axis=0)
                closest_cores = distance_to_core.argmin(axis=1)
                distance_to_core = distance_to_core[np.arange(distance_to_core.shape[0]),
                                                    closest_cores]
                # No cluster by default.
                small_labels = -np.ones(distance_to_core.shape[0])
                # Label only the ones with distance less than eps.
                within_range = distance_to_core <= eps**2
                small_labels[within_range] =\
                    large_labels[core_indices[closest_cores[within_range]]]

                # Assign labels to stixels. Changing instance_stixels also changes
                # augmented_stixels.
                labels[large_indices] = large_labels
                labels[small_indices] = small_labels
                labels[labels >= 0] = labels[labels >= 0] + class_*1000

        for idx, label in enumerate(labels):
            instance_stixels[idx]['instance_label'] = int(label)

        # Collect centers for to overlay in visualization.
        if collect_centers:
            labels = labels.astype(np.int)
            unique_labels = set(labels)

            for label in unique_labels:
                instance_color =\
                        COLOR_LUT_RANDOM[label % len(COLOR_LUT_RANDOM),:]
                if label == -1:
                    instance_color = np.array((255,255,255))
                instance_color = instance_color.astype(np.float)
                collected_centers.append((instance_color, X[:,labels==label]))

    if collect_centers:
        return augmented_stixels, collected_centers
    return augmented_stixels

def overlay_centers(image, collected_centers):
    """
    image : cv2 image
    collected_centers : list of elements [color, center_coords[2,:]]
    """
    for color, centers in collected_centers:
        for center in centers.transpose():
            border_color = (255, 255, 255)
            if np.all(color == 255):
                border_color = (0,0,0)
            center = (int(center[0]), int(image.shape[0]-center[1]))
            cv2.circle(image, center, 5, color, thickness=-1)
            cv2.circle(image, center, 5, border_color, thickness=1)
    return image

def add_instance_disparity(stixels, instance_disparity):
    img_shape = instance_disparity.shape
    # Let's not change the stixel list in place, but make a copy.
    instance_disparity_stixels = copy.deepcopy(stixels)
    stixel_width = img_shape[1] // len(stixels)

    for col, column_stixels in enumerate(instance_disparity_stixels):
        for stixel in column_stixels:
            topleft_x = col * stixel_width
            topleft_y = img_shape[0] - stixel['vT'] - 1 # vT, mirror y-axis
            bottomright_x = topleft_x + stixel_width - 1 # e.g. width 5: 0-4,5-9,
            bottomright_y = img_shape[0] - stixel['vB'] - 1 # vB, mirror y-axis

            disparities = instance_disparity[topleft_y:bottomright_y+1,
                                             topleft_x:bottomright_x+1]
            #disparities = disparities.astype(np.uint8)
            disparities[disparities < 1] = 0
            non_zero_disparities = disparities[disparities != 0]

            # compute median disparity in stixel
            # OR: compute majority-vote disparity in stixel
            if len(non_zero_disparities) > 0 and stixel['class'] > 10:
                stixel['instance_disparity'] = np.median(non_zero_disparities)
            else:
                stixel['instance_disparity'] = 0

    return instance_disparity_stixels

def compute_instance_disparity(disparity, instance_masks):
    """
    Compute median disparity per mask and assign it to every pixel in mask.
    """
    instance_disparity = np.zeros(disparity.shape) # h x w

    # Get all relevant instance labels (id > 1000).
    for instance_mask in instance_masks:
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids > 1000]

        for instance_id in instance_ids:
            binary_mask = (instance_mask == instance_id)
            # Note: this median behaves differently than the pytorch median.
            # See: https://github.com/torch/torch7/pull/182
            disparities = disparity[binary_mask].astype(np.float)
            non_zero_disparities = disparities[disparities != 0]

            if len(non_zero_disparities) > 0:
                median_disparity = np.median(non_zero_disparities)
            else:
                median_disparity = 0

            instance_disparity[binary_mask] = median_disparity

    return instance_disparity

def process_stixelfile(filepaths, max_disparity, output_directory, topdown,
        resultsonly, cluster_config):
    filename_base = os.path.basename(filepaths['stixel'])
    filename_base = filename_base[:-8] # remove ".stixels"
    stixels, groundplane = read_stixel_file(filepaths['stixel'])
    #print("len(stixels) = {}".format(len(stixels)))
    #print("len(stixels[0][0]) = {}".format(len(stixels[0][0])))

    image = cv2.imread(filepaths['image'])
    print("image.max() = {}".format(image.max()))
    if os.path.exists(filepaths['camera']):
        with open(filepaths['camera'], 'r') as camera_file:
            camera_parameters = json.load(camera_file)
    else:
        camera_parameters = None


    # load instance ground truth if available
    instancegt_mask = None
    if 'instance' in filepaths.keys():
        instancegt_mask = load_instance_mask(filepaths['instance'])

    if 'disparity' in filepaths.keys():
        original_disparity =\
                cv2.imread(filepaths['disparity'], cv2.IMREAD_GRAYSCALE)
        instance_disparity =\
                compute_instance_disparity(original_disparity, instancegt_mask)
        stixels = add_instance_disparity(stixels, instance_disparity)
    if 'probs' in filepaths.keys():
        with h5.File(filepaths['probs']) as h5f:
            instance_disparity = h5f['nlogprobs'][:,:,19]
        stixels = add_instance_disparity(stixels, instance_disparity)

    collected_centers = []
    assigned_stixels = None
    if 'instance_mean_x' in stixels[0][0].keys():
        if not cluster_config['use_instancegt']:
            assigned_stixels, collected_centers =\
                    assign_instances(stixels,
                                     cluster_config,
                                     collect_centers=True,
                                     camera_parameters=camera_parameters)
            if len(cluster_config['use_instance_disparity']) > 0:
                fig = plt.figure()
                for color, centers in collected_centers:
                    plt.scatter(centers[0,:], centers[2,:],
                                c=np.atleast_2d(color/255), s=1)
                    plt.axis('equal')
                    plt.grid(True)
                plt.ylim(0,90)
                plt.xlim(0,90)
                fig.savefig(os.path.join(output_directory, "stixelsim",
                                         #filename_base+'_topdown{}.pdf'.format(i)),
                                         filename_base+'_topdownclustering.png'),
                                         dpi=300,
                                         bbox_inches='tight')
                                         #filename_base+'_topdown.pdf'))
                plt.close(fig)
        else:
            print("Using ground truth to assign instances.")
            assigned_stixels = assign_instances_gt(stixels, instancegt_mask)
            # Don't draw/assign them again in draw stixels.
            instancegt_mask = None
    else:
        raise ValueError("No instance information for some reason.")

    # plot top down
    if topdown:
        tmp_stixels = stixels if assigned_stixels is None else assigned_stixels
        pc = pointcloud(tmp_stixels, image.shape, max_disparity, groundplane,
                        camera_parameters, instancegt_mask)
        #filter_mask = lambda x: x[:,1] == x[:,1]
        filter_mask = lambda x: (x[:,2] < 120.0)\
                              & (x[:,1] < 5.0) & (x[:,1] > -5.0)

        size_filter = cluster_config['size_filter']
        figs = plot_topdownview(pc, filter_mask, size_filter)
        for i,fig in enumerate(figs):
            fig.savefig(os.path.join(output_directory, "stixelsim",
                                     filename_base+'_topdown{}.png'.format(i)),
                                     #filename_base+'_topdown{}.svg'.format(i)),
                                     dpi=400,
                                     bbox_inches='tight')
                                     #filename_base+'_topdown.pdf'))
            plt.close(fig)

    contour_image = None
    if True: # save instance masks
        masks = draw_instance_masks(assigned_stixels, image.shape[:2])
        instance_mask_dir =\
                os.path.join(output_directory, "results", "instance_preds")
        if not os.path.isdir(instance_mask_dir):
            try: # needed because of race condition when multiprocessing
                os.makedirs(instance_mask_dir)
            except OSError as e:
                if e.errno != os.errno.EEXIST:
                    raise e
        save_instance_masks(instance_mask_dir, filename_base,
                            masks)
        if True: # draw contour image
            contour_image = np.zeros((*image.shape[:2],3), dtype=np.uint8)
            for id_, mask in masks.items():
                color = trainId2color[id_ // 1000]
                color = np.array(color[::-1]) # RGB2BGR
                color[:] = 255
                _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
                contour_image = cv2.drawContours(contour_image, contours, -1,
                                                 color.tolist(), 2)

    disparity_image = None
    disparity_result_image = None
    semantic_image = None
    cost_image = None
    if not resultsonly:
        disparity_image = np.zeros_like(image, dtype=np.uint8)
        disparity_result_image = np.zeros(image.shape[:2], dtype=np.float)
        semantic_image = np.zeros(image.shape, dtype=np.uint8)\
                         if 'class' in stixels[0][0].keys() else None
        cost_image = np.zeros(image.shape[:2], dtype=np.uint8)\
                     if 'cost' in stixels[0][0].keys() else None
    semantic_labelImg = np.zeros(image.shape[:2], dtype=np.uint8)\
                        if 'class' in stixels[0][0].keys() else None
    instance_image = np.zeros((*image.shape[:2],3), dtype=np.uint8)\
                     if 'instance_mean_x' in stixels[0][0].keys() else None
    median_disparity_image = np.zeros(image.shape[:2], dtype=np.uint8)\
            if 'instance_disparity' in stixels[0][0].keys() else None

    # You can not imagine how much I shiver when I look at this line. Not
    # because it is wrong. Just because it's ugly.
    (disparity_image, semantic_image, semantic_labelImg, cost_image,
     stixel_count, instance_gt_image, disparity_result_image, instance_image,
     median_disparity_image) =\
            draw_stixels(assigned_stixels, max_disparity, disparity_image,
                         semantic_image, semantic_labelImg, cost_image,
                         instancegt_mask=instancegt_mask,
                         disparity_result_image=disparity_result_image,
                         instance_image=instance_image,
                         median_disparity_image=median_disparity_image)

    if semantic_image is not None:
        #semantic_image = np.dstack([semantic_image,
        #                            semantic_image,
        #                            semantic_image]).astype(np.uint8)
        #semantic_image_color = cv2.LUT(semantic_image, COLOR_LUT)

        # Save raw.
        cv2.imwrite(os.path.join(output_directory, "stixelsim",
                                 filename_base+'_segmentationstixel.png'),
                    semantic_image)
        # Save overlay of images.
        alpha = 0.3
        semantic_image = cv2.addWeighted(image, alpha, semantic_image, 1.0-alpha, 0)
        cv2.imwrite(os.path.join(output_directory, "stixelsim",
                                 filename_base+'_segmentationstixeloverlay.png'),
                    semantic_image)

    if contour_image is not None:
        zero_mask = (contour_image == 0).all(axis=2)
        if semantic_image is not None:
            contour_image[zero_mask] = semantic_image[zero_mask]
        else:
            contour_image[zero_mask] = image[zero_mask]
        cv2.imwrite(os.path.join(output_directory, "stixelsim",
                                 filename_base+'_instancecontour.png'),
                    contour_image)

    if semantic_labelImg is not None:
        cv2.imwrite(os.path.join(output_directory, "results/preds",
                                 filename_base+'_labelImg.png'),
                    semantic_labelImg)

    if disparity_result_image is not None:
        # TODO: Do not trust this result. Ground stixels are not handled
        # approroately. Also, PIL export and import do not match to 100%. Mean
        # value differs.

        # Rescale to original disparity range.
        disparity_result_image *= 256.0
        im = PIL.Image.fromarray(disparity_result_image.astype(np.float32))
        im.save(os.path.join(output_directory, "results/disparity",
                             filename_base+'_disparity.tiff'))

    if cost_image is not None:
        cv2.imwrite(os.path.join(output_directory, "stixelsim",
                                 filename_base+'_coststixel.png'),
                    cost_image)

    if instance_image is not None:
        # Collected clusters will be in x,y,z when using instance_disparity.
        if len(cluster_config['use_instance_disparity']) == 0:
            instance_image = overlay_centers(instance_image, collected_centers)
        instance_image = cv2.cvtColor(instance_image.astype(np.uint8),
                                      cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_directory, "stixelsim",
                                 filename_base+'_instancestixel.png'),
                    instance_image)

    if instance_gt_image is not None:
        instance_gt_image = cv2.cvtColor(instance_gt_image.astype(np.uint8),
                                         cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(os.path.join(output_directory, "stixelsim",
                                 filename_base+'_instancegtstixel.png'),
                    instance_gt_image)

    if median_disparity_image is not None:
        # Save raw.
        cv2.imwrite(os.path.join(output_directory, "stixelsim",
                                 filename_base+'_mediandisparitystixel.png'),
                    median_disparity_image)
    if disparity_image is not None:
        # Save raw.
        cv2.imwrite(os.path.join(output_directory, "stixelsim",
                                 filename_base+'_disparitystixel.png'),
                    disparity_image)
        # Save overlay of images.
        alpha = 0.3
        disparity_image = cv2.addWeighted(image, alpha, disparity_image, 1.0-alpha, 0)
        # Draw horizon from stixels. groundplane[1] starts counting from
        # bottom. Thus, mirror it.
        cv2.rectangle(disparity_image,
                      (0, disparity_image.shape[0]-1-groundplane[1]),
                      (disparity_image.shape[1], disparity_image.shape[0]-1-groundplane[1]),
                      (0,0,0),
                      thickness=-1) # fill rectangle
        cv2.imwrite(os.path.join(output_directory, "stixelsim",
                                 filename_base+'_disparitystixeloverlay.png'),
                    disparity_image)

    # Plot and save.
    #disparity_fig = plt.figure()
    #plt.axis('off')
    #plt.imshow(disparity_image[:,:,::-1]) # opencv is BGR instead of RGB
    #plt.tight_layout()
    #disparity_fig.savefig(
    #    os.path.join(output_directory, filename_base+'_disparitystixel.png'))

    #segmentation_fig = plt.figure()
    #plt.axis('off')
    #plt.imshow(semantic_image[:,:,::-1]) # opencv is BGR instead of RGB
    #plt.tight_layout()
    #segmentation_fig.savefig(
    #    os.path.join(output_directory, filename_base+'_segmentationstixel.png'))
    return stixel_count

def process_imagefile(image_file, stixel_dir, image_dir, max_disparity,
        data_directory, topdown, resultsonly, cluster_config):
    total_stixels = 0
    no_files = 0
    filename_base = "_".join(image_file.split("_")[:-1])
    print("Looking for files {}".format(os.path.join(stixel_dir,
                                                     filename_base+'*')))
    for stixel_file in glob.iglob(os.path.join(stixel_dir,
                                               filename_base+'*')):
        # --- Setup filepaths.
        filepaths = {'stixel' : stixel_file}
        filepaths['image'] = os.path.join(image_dir, image_file)

        if (cluster_config['use_instancegt']
            or len(cluster_config['use_instance_disparity']) > 0):
            # Get instance gt files.
            filepaths['instance'] = os.path.join(data_dir, "gtFine",
                    filename_base + "_gtFine_instanceIds.png")
            if not os.path.exists(filepaths['instance']):
                raise IOError("Could not find instance mask at {}"
                              .format(filepaths['instance']))

        if cluster_config['use_instance_disparity'] == "from_gt":
            # Get disparity files.
            filepaths['disparity'] = os.path.join(data_dir, "disparities",
                    filename_base + "_disparity.png")
            if not os.path.exists(filepaths['disparity']):
                raise IOError("Could not find disparity ground truth at {}"
                              .format(filepaths['disparity']))

        if cluster_config['use_instance_disparity'] == "from_pred":
            # Get disparity files.
            filepaths['probs'] = os.path.join(data_dir, "probs",
                    filename_base + "_probs.h5")
            if not os.path.exists(filepaths['probs']):
                raise IOError("Could not find cnn prediction at {}"
                              .format(filepaths['probs']))

        filepaths['camera'] = os.path.join(data_dir, "camera",
                filename_base + "_camera.json")

        print("Processing {} and {}.".format(stixel_file, image_file))
        stixel_count = process_stixelfile(filepaths, max_disparity,
                                          data_directory, topdown,
                                          resultsonly, cluster_config)
        total_stixels += stixel_count
        no_files += 1
    return total_stixels, no_files

def main(stixel_dir, image_dir, max_disparity, data_directory, topdown,
         resultsonly, cluster_config):
    args = tuple((image_file, stixel_dir, image_dir, max_disparity,
                  data_directory, topdown, resultsonly, cluster_config)
                 for image_file in os.listdir(image_dir))
    if 'ipdb' not in sys.modules:
        with multiprocessing.Pool(processes=None) as pool:
            results = pool.starmap(process_imagefile, args)
    else: # process sequentially
        results = itertools.starmap(process_imagefile, args)
    total_stixels, no_files = zip(*results)
    total_stixels = sum(total_stixels)
    no_files = sum(no_files)

    print("")
    print("Total number of stixels = {}".format(total_stixels))
    print("Total number of files = {}".format(no_files))
    print("Average number of stixels per image = {}".\
          format(total_stixels / float(no_files)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stixel visualization.")
    parser.add_argument("--topdown","-t", action='store_true',
                        help="Save top down view.")
    parser.add_argument("--use-instancegt","-i", action='store_true',
                        help="Use ground truth to assign instances.")
    parser.add_argument("--use-disparity", type=str, default="",
                        help="Use instance disparity for clustering. "
                        "Options: from_gt, from_pred")
    parser.add_argument("--maxdisparity","-d", default=128, type=int,
                        help="Max disparity.")
    parser.add_argument("--resultsonly","-r", action='store_true',
                        help="Save only files required for evaluation.")
    parser.add_argument("--eps", type=float, default=16,
                        help="Max distance for core neighborhood in DBSCAN.")
    parser.add_argument("--min_size", type=int, default=2,
                        help="Minimum size for core neighborhood in DBSCAN.")
    parser.add_argument("--size_filter", type=int, default=10,
                        help="Minimum number of rows in stixel to be core "\
                        "point candidate in DBSCAN.")
    parser.add_argument("DATADIR", type=str,
                        help="Data directory directory.")
    #type=str,
    #choices=['CIFAR10','MNIST'],
    #default=[2], type=int, nargs='+',
    #action='store_true',
    args = parser.parse_args()

    data_dir = args.DATADIR
    #output_directory = os.path.expanduser(args.outdir)
    #output_directory = os.path.abspath(output_directory)
    for directory in ['stixelsim', 'results/preds','results/disparity']:
        directory = os.path.join(data_dir, directory)
        if not os.path.isdir(directory):
            os.makedirs(directory)

    stixel_dir = os.path.join(data_dir, "stixels")
    image_dir = os.path.join(data_dir, "left")

    cluster_config = {
            'eps'                   : args.eps,
            'min_size'              : args.min_size,
            'size_filter'           : args.size_filter,
            'use_instancegt'        : args.use_instancegt,
            'use_instance_disparity': args.use_disparity }

    main(stixel_dir, image_dir, args.maxdisparity, data_dir, args.topdown,
         args.resultsonly, cluster_config)
