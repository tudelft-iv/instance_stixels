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
import os
import glob
import json
import copy

import numpy as np
# TODO: Probably cv2 should be replaced by pillow, which should provide the
# same functionality and is necessary to save 32bit float tiff files. Also, it
# might provide better error messages.
import cv2
import PIL.Image
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

import cityscapesscripts.helpers.labels as cityscapes_labels
from cityscapes_instance_loader import load_instance_mask

# --- Color look up tables
trainId2color = [label.color for label in cityscapes_labels.labels\
                 if label.trainId not in (-1,255)]

COLOR_LUT = np.zeros((256,1,3), dtype=np.uint8)
# COLOR_LUT[0] is used for stixel frame -> (0,0,0)
COLOR_LUT[0,0,:] = 255
COLOR_LUT[1:20,0,:] = np.array(trainId2color).astype(np.uint8)
COLOR_LUT = cv2.cvtColor(COLOR_LUT, cv2.COLOR_RGB2BGR)

COLOR_LUT_RANDOM = np.random.randint(255, size=(217,3))

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
                if len(stixel) == 4: # no semantic information
                    stixels[-1].append([int(stixel[0]), # type
                                        int(stixel[1]), # vB
                                        int(stixel[2]), # vT
                                        float(stixel[3])]) # disparity
                elif len(stixel) == 5: # with semantic information
                    stixels[-1].append([int(stixel[0]), # type
                                        int(stixel[1]), # vB
                                        int(stixel[2]), # vT
                                        float(stixel[3]), # disparity
                                        int(stixel[4])]) # class
                elif len(stixel) == 6: # with semantic and cost information
                    stixels[-1].append([int(stixel[0]), # type
                                        int(stixel[1]), # vB
                                        int(stixel[2]), # vT
                                        float(stixel[3]), # disparity
                                        int(stixel[4]), # class
                                        float(stixel[5])]) # cost
                elif len(stixel) == 8: # with semantic, cost and instance mean
                    stixels[-1].append([int(stixel[0]), # type
                                        int(stixel[1]), # vB
                                        int(stixel[2]), # vT
                                        float(stixel[3]), # disparity
                                        int(stixel[4]), # class
                                        float(stixel[5]), # cost
                                        float(stixel[6]), # instance_mean_x
                                        float(stixel[7])]) # instance_mean_y
                else:
                    raise IOError("Invalid length of stixel description "
                                  "len(stixel = {}) = {}.".\
                                     format(stixel, len(stixel)))
    return stixels, groundplane

def draw_instance_masks(stixels, mask_shape):
    stixel_width = mask_shape[1] // len(stixels)
    masks = {}
    for col, column_stixels in enumerate(stixels):
        for stixel in column_stixels:
            if len(stixel) > 8 and stixel[8] > 0:
                instance_id = stixel[8]
                if instance_id not in masks.keys():
                    masks[instance_id] = np.zeros(mask_shape, dtype=np.uint8)
                instance_mask = masks[instance_id]

                topleft_x = col * stixel_width
                topleft_y = mask_shape[0] - stixel[2] - 1 # vT, mirror y-axis
                bottomright_x = topleft_x + stixel_width - 1 # e.g. width 5: 0-4,5-9,
                bottomright_y = mask_shape[0] - stixel[1] - 1 # vB, mirror y-axis

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
            raise ValueError("These classes should not have instances. "\
                             "Class id = {}.".format(class_id))
        txtlines.append("{} {} 1.0".format(mask_file, class_id))

    with open("{}/{}.txt".format(instance_mask_dir, filename_base), 'w') as f:
        for line in txtlines:
            f.write(line+'\n')

def draw_stixels(stixels, max_disparity,
        disparity_image=None, semantic_image=None, semantic_labelImg=None,
        cost_image=None, max_cost=1e4, instancegt_mask=None,
        disparity_result_image=None, instance_image=None):
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
            factor = max((stixel[5] - min_cost),0) / (max_cost - min_cost)
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
            topleft_y = shape[0] - stixel[2] - 1 # vT, mirror y-axis
            bottomright_x = topleft_x + stixel_width - 1 # e.g. width 5: 0-4,5-9,
            bottomright_y = shape[0] - stixel[1] - 1 # vB, mirror y-axis

            # type distinction
            # type 2 (sky)
            if stixel[0] == 2 or (stixel[0] == 1 and stixel[3] < 1.0):
                color = (255,191,0)
            elif stixel[0] == 1: # type 1 (object)
                # Cropping of at disparities of max_disparity * 2e-2 = 2.56 and
                # below.
                distance = min(float(stixel[3]+20)/max_disparity, 1.)
                color = np.array([[cm.inferno(distance)[:3]]])*255
                color = color.astype(np.uint8)
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                color = color[0,0].tolist()

                # compute instance affiliation from ground truth
                if instancegt_mask is not None:
                    # Only consider classes with instances.
                    if stixel[4] < 11: # 10 is sky
                        instance_color = np.array((0.,0.,0.,255.))
                    else:
                        max_instanceid = 0
                        # Must be at least 10% of stixel.
                        max_instanceid_pixels = 0
                        # Only use instance mask of predicted class.
                        # Before:for mask in instancegt_mask:
                        mask = instancegt_mask[stixel[4]-11]
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
                        if max_instanceid == 0:
                            instance_color[:3] = 255.0
                            instance_color[3] = 255.0
                        elif (max_instanceid_pixels <
                                0.1*stixel_width*(bottomright_y-topleft_y)):
                            instance_color[:3] = 255.0

            if instancegt_mask is not None:
                if instance_gt_image is None:
                    instance_gt_image = np.zeros((*disparity_image.shape[:2],4))
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
                if len(stixel) > 8:
                    instance_id = int(stixel[8])
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
                if True:
                    if np.abs(topleft_y - bottomright_y) > 1:
                        cv2.rectangle(instance_image,
                                      (topleft_x, topleft_y), 
                                      (bottomright_x, topleft_y-2),
                                      (255,255,255),
                                      thickness=-1) # border only on left side
            if disparity_image is not None and len(stixel) > 4:
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
            if semantic_image is not None and len(stixel) > 4:
                cv2.rectangle(semantic_image,
                              (topleft_x, topleft_y), 
                              (bottomright_x, bottomright_y),
                              stixel[4]+1,
                              thickness=-1) # fill rectangle
                if False: # one pixel width border (around)
                    if np.abs(topleft_y - bottomright_y) > 1:
                        cv2.rectangle(semantic_image,
                                      (topleft_x, topleft_y), 
                                      (bottomright_x+1, bottomright_y+1),
                                      0,
                                      thickness=1) # border only on left side
                if True: # upper end border of "width" pixels
                    width = 1
                    if np.abs(topleft_y - bottomright_y) > 1:
                        cv2.rectangle(semantic_image,
                                      (topleft_x, topleft_y), 
                                      (bottomright_x, topleft_y-width),
                                      0,
                                      thickness=-1) # border only on left side
            if cost_image is not None and len(stixel) > 4:
                cv2.rectangle(cost_image,
                              (topleft_x, topleft_y), 
                              (bottomright_x, bottomright_y),
                              int(cost),
                              thickness=-1) # fill rectangle
            # --- result images
            if semantic_labelImg is not None and len(stixel) > 4:
                cv2.rectangle(semantic_labelImg,
                              (topleft_x, topleft_y), 
                              (bottomright_x, bottomright_y),
                              cityscapes_labels.trainId2label[stixel[4]].id,
                              thickness=-1) # fill rectangle
            if disparity_result_image is not None:
                # TODO: Handle ground stixels appropriately.
                cv2.rectangle(disparity_result_image,
                              (topleft_x, topleft_y), 
                              (bottomright_x, bottomright_y),
                              stixel[3],
                              thickness=-1) # fill rectangle
    print("Processed {} stixels.".format(stixel_count))
    return (disparity_image, semantic_image, semantic_labelImg, cost_image,
            stixel_count, instance_gt_image, disparity_result_image,
            instance_image)

def get_instance_means(stixels, class_=None):
    x_instance_pos = []
    y_instance_pos = []
    size_instance = []
    instance_stixels = []
    for col, column_stixels in enumerate(stixels):
        for stixel in column_stixels:
            # Only consider stixel with instance class.
            if stixel[4] > 10 and (class_ is None or stixel[4] == class_):
                x_instance_pos.append(stixel[6])
                y_instance_pos.append(stixel[7])
                size_instance.append(stixel[2] - stixel[1] + 1)
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
            if stixel[4] < 11:
                continue

            max_instanceid = -1
            max_instanceid_pixels = 0
            # Only use instance mask of predicted class.
            # Before: for mask in instancegt_mask:
            mask = instancegt_mask[stixel[4]-11]

            topleft_x = col * stixel_width
            topleft_y = mask_shape[0] - stixel[2] - 1 # vT, mirror y-axis
            bottomright_x = topleft_x + stixel_width - 1 # e.g. width 5: 0-4,5-9,
            bottomright_y = mask_shape[0] - stixel[1] - 1 # vB, mirror y-axis

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
                max_instanceid = max_instanceid % 1000 + stixel[4] * 1000
            else:
                max_instanceid = -1
            augmented_stixels[col][s_idx].append(max_instanceid)

    return augmented_stixels

def assign_instances(stixels, eps=16, min_samples=2, size_filter=10):
    augmented_stixels = copy.deepcopy(stixels)
    for class_ in range(11,19):
        X, size, instance_stixels =\
                get_instance_means(augmented_stixels, class_)
        # Filter small stixels.
        large_mask = size >= size_filter
        large_indices = np.nonzero(large_mask)[0]
        X_large = X[:, large_mask]

        small_mask = size < size_filter
        small_indices = np.nonzero(small_mask)[0]
        X_small = X[:, small_mask]

        labels = -np.ones(X.shape[1])
        # Given we have a couple of large stixels of that class.
        if X_large.shape[1] > min_samples:
            # Cluster.
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
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
            instance_stixels[idx].append(int(label))

    return augmented_stixels

def process_file(filepaths, max_disparity, output_directory,
        resultsonly, cluster_config):
    filename_base = os.path.basename(filepaths['stixel'])
    filename_base = filename_base[:-8] # remove ".stixels"
    stixels, groundplane = read_stixel_file(filepaths['stixel'])

    image = cv2.imread(filepaths['image'])
    print("image.max() = {}".format(image.max()))

    # load instance ground truth if available
    instancegt_mask = None
    if 'instance' in filepaths.keys():
        instancegt_mask = load_instance_mask(filepaths['instance'])

    assigned_stixels = None
    if len(stixels[0][0]) >= 8:
        if instancegt_mask is None:
            assigned_stixels = assign_instances(
                                    stixels,
                                    cluster_config['eps'],
                                    cluster_config['min_size'],
                                    cluster_config['size_filter'])
        else:
            print("Using ground truth to assign instances.")
            assigned_stixels = assign_instances_gt(stixels, instancegt_mask)
            # Don't draw/assign them again in draw stixels.
            instancegt_mask = None
    else:
        raise ValueError("No instance information for some reason.")

    contour_image = None
    if True: # save instance masks
        masks = draw_instance_masks(assigned_stixels, image.shape[:2])
        instance_mask_dir =\
                os.path.join(output_directory, "results", "instance_preds")
        if not os.path.isdir(instance_mask_dir):
            os.makedirs(instance_mask_dir)
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
        semantic_image = np.zeros(image.shape[:2], dtype=np.uint8)\
                         if len(stixels[0][0]) >= 5 else None
        cost_image = np.zeros(image.shape[:2], dtype=np.uint8)\
                     if len(stixels[0][0]) >= 6 else None
    semantic_labelImg = np.zeros(image.shape[:2], dtype=np.uint8)\
                        if len(stixels[0][0]) >= 5 else None
    instance_image = np.zeros((*image.shape[:2],3), dtype=np.uint8)\
                     if len(stixels[0][0]) >= 5 else None

    # You can not imagine how much I shiver when I look at this line. Not
    # because it is wrong. Just because it's ugly.
    disparity_image, semantic_image, semantic_labelImg, cost_image,\
    stixel_count, instance_gt_image, disparity_result_image, instance_image =\
            draw_stixels(assigned_stixels, max_disparity, disparity_image,
                         semantic_image, semantic_labelImg, cost_image,
                         instancegt_mask=instancegt_mask,
                         disparity_result_image=disparity_result_image,
                         instance_image=instance_image)

    semantic_image_color = None
    if semantic_image is not None:
        semantic_image = np.dstack([semantic_image, 
                                    semantic_image,
                                    semantic_image]).astype(np.uint8)
        semantic_image_color = cv2.LUT(semantic_image, COLOR_LUT)

        # Save raw.
        cv2.imwrite(os.path.join(output_directory, "stixelsim",
                                 filename_base+'_segmentationstixel.png'),
                    semantic_image_color)
        # Save overlay of images.
        alpha = 0.3
        semantic_image = cv2.addWeighted(image, alpha, semantic_image_color, 1.0-alpha, 0)
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

    if cost_image is not None:
        cv2.imwrite(os.path.join(output_directory, "stixelsim",
                                 filename_base+'_coststixel.png'),
                    cost_image)

    if instance_image is not None:
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

    return stixel_count

def main(stixel_dir, image_dir, max_disparity, data_directory, 
         use_instancegt, resultsonly, cluster_config):
    total_stixels = 0
    no_files = 0
    for image_file in os.listdir(image_dir):
        filename_base = "_".join(image_file.split("_")[:-1])
        print("Looking for files {}".format(os.path.join(stixel_dir, 
                                                         filename_base+'*')))
        for stixel_file in glob.iglob(os.path.join(stixel_dir, 
                                                   filename_base+'*')):
            # --- Setup filepaths.
            filepaths = {'stixel' : stixel_file}
            filepaths['image'] = os.path.join(image_dir, image_file)

            if use_instancegt:
                filepaths['instance'] = os.path.join(data_dir, "gtFine",
                        filename_base + "_gtFine_instanceIds.png")
                if not os.path.exists(filepaths['instance']):
                    raise IOError("Could not find instance mask at {}"
                                  .format(filepaths['instance']))

            filepaths['camera'] = os.path.join(data_dir, "camera",
                    filename_base + "_camera.json")

            print("Processing {} and {}.".format(stixel_file, image_file))
            stixel_count = process_file(filepaths, max_disparity,
                                        data_directory, resultsonly,
                                        cluster_config)
            total_stixels += stixel_count
            no_files += 1
    print("")
    print("Total number of stixels = {}".format(total_stixels))
    print("Total number of files = {}".format(no_files))
    print("Average number of stixels per image = {}".\
          format(total_stixels / float(no_files)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stixel visualization.")
    parser.add_argument("--use-instancegt","-i", action='store_true',
                        help="Use ground truth to assign instances.")
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
    args = parser.parse_args()

    data_dir = args.DATADIR
    for directory in ['stixelsim', 'results/preds','results/disparity']:
        directory = os.path.join(data_dir, directory)
        if not os.path.isdir(directory):
            os.makedirs(directory)

    stixel_dir = os.path.join(data_dir, "stixels")
    image_dir = os.path.join(data_dir, "left")

    cluster_config = {
            'eps'           : args.eps,
            'min_size'      : args.min_size,
            'size_filter'   : args.size_filter}

    main(stixel_dir, image_dir, args.maxdisparity, data_dir,
         args.use_instancegt, args.resultsonly, cluster_config)
