# This file is part of Instance Stixels:
# https://github.com/tudelft-iv/instance-stixels
#
# Copyright (c) 2019 Thomas Hehn.
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
# along with Instance Stixels. If not, see <http://www.gnu.org/licenses/>.

import torch
from torch import nn

#import ipdb

class DisparityOffsetLossSL:
    def __init__(self, offset_mean_weight=1e-3, offset_variance_weight=1e-4,
                 disparity_mean_weight=1e-3, disparity_variance_weight=1e-4,
                 abs_variance=False):
        self.weights = {
                'offset_mean' : offset_mean_weight,
                'offset_variance' : offset_variance_weight,
                'disparity_mean' : disparity_mean_weight,
                'disparity_variance' : disparity_variance_weight }
        self.abs_variance = abs_variance

    def __call__(self, batch_prediction, batch_instance_gt,
            batch_disparity_gt, separate=False):
        offset_mean_loss = 0
        offset_variance_loss = 0
        disparity_mean_loss = 0
        disparity_variance_loss = 0

        for instance_gt, disparity_img, prediction in\
                zip(batch_instance_gt, batch_disparity_gt, batch_prediction):
            disparity_img = disparity_img.squeeze()
            # --- Instance pixels.
            instance_ids = torch.unique(instance_gt)
            # Get all relevant instance labels (id > 1000).
            instance_ids = instance_ids[instance_ids > 1000]
            for instance_id in instance_ids:
                binary_mask = (instance_gt.squeeze() == instance_id)
                mask_ind = torch.nonzero(binary_mask).t()
                n_pixels = mask_ind.size(1)
                mask_positions = mask_ind.float()

                # --- Offsets.
                pred_positions = (prediction[1:,mask_ind[0],mask_ind[1]]
                                  + mask_positions)

                gt_mean = mask_positions.mean(dim=1).reshape(2,1).detach()
                # TODO: use **2 instead?
                offset_mean_loss +=\
                        (pred_positions-gt_mean).abs().sum() / n_pixels / 2

                # Note: there is an instance label in the cropped dataset, that
                # has a single pixel instance annotation. Be careful with var()
                # in this case.
                if not self.abs_variance:
                    offset_variance_loss +=\
                        pred_positions.var(dim=1, unbiased=False).sum() / 2
                else:
                    if n_pixels > 2:
                        pred_mean = pred_positions.mean(dim=1).reshape(2,1)
                        offset_variance_loss +=\
                            (pred_positions - pred_mean).abs().sum()\
                            / n_pixels / 2
                        del pred_mean


                # --- Disparity.
                pred_disparities = prediction[0,mask_ind[0],mask_ind[1]]
                gt_disparities = disparity_img[mask_ind[0],mask_ind[1]]
                # Ignore invalid disparities.
                gt_disparities = gt_disparities[gt_disparities != 0]

                if not self.abs_variance:
                    disparity_variance_loss +=\
                        pred_disparities.var(unbiased=False)
                else:
                    if n_pixels > 2:
                        pred_mean = pred_disparities.mean()
                        disparity_variance_loss +=\
                            (pred_disparities - pred_mean).abs().sum() / n_pixels
                        del pred_mean

                if len(gt_disparities) > 0:
                    gt_median = gt_disparities.median()
                    disparity_mean_loss +=\
                        (pred_disparities-gt_median).abs().sum() / n_pixels

            # --- Non-instance pixels.
            # (id < 11) + (id == 255):
            # Do not penalize any pixel that has semantic instance
            # class (e.g. car), but have no instance information.
            # Alternative: (id < 1000).
            binary_mask = ((instance_gt.squeeze() < 11)
                           + (instance_gt.squeeze() == 255))
            n_pixels = binary_mask.sum()
            offset_mean_loss +=\
                    prediction[1:,binary_mask].abs().sum() / n_pixels / 2
            disparity_mean_loss +=\
                    prediction[0,binary_mask].abs().sum() / n_pixels

        # Useful prints for loss balancing.
        #print("offset_mean_loss =",offset_mean_loss)
        #print("offset_variance_loss =",offset_variance_loss)
        #print("disparity_mean_loss =",disparity_mean_loss)
        #print("disparity_variance_loss =",disparity_variance_loss)
        loss = (self.weights['offset_mean'] * offset_mean_loss
                + self.weights['offset_variance'] * offset_variance_loss
                + self.weights['disparity_mean'] * disparity_mean_loss
                + self.weights['disparity_variance'] * disparity_variance_loss)
        if not separate:
            return loss
        return torch.Tensor((loss, offset_mean_loss, offset_variance_loss,
                             disparity_mean_loss, disparity_variance_loss))

class OffsetLossSL:
    def __init__(self, offset_mean_weight=1e-3, offset_variance_weight=1e-4,
                 **kwargs):
        #self.mean_loss_fct = nn.MSELoss(reduction='mean')
        self.mean_weight = offset_mean_weight
        self.variance_weight = offset_variance_weight

    def __call__(self, batch_prediction, batch_instance_gt):
        mean_loss = 0
        variance_loss = 0

        for instance_gt, prediction in zip(batch_instance_gt, batch_prediction):
            # --- Instance pixels.
            instance_ids = torch.unique(instance_gt)
            # Get all relevant instance labels (id > 1000).
            instance_ids = instance_ids[instance_ids > 1000]
            for instance_id in instance_ids:
                binary_mask = (instance_gt.squeeze() == instance_id)
                mask_ind = torch.nonzero(binary_mask).t()
                n_offsets = mask_ind.size(1) * 2
                mask_positions = mask_ind.float()

                pred_positions = (prediction[:,mask_ind[0],mask_ind[1]]
                                  + mask_positions)
                # Note: there is an instance label in the cropped dataset, that
                # has a single pixel instance annotation. Be careful with var()
                # in this case.
                variance_loss +=\
                    pred_positions.var(dim=1, unbiased=False).sum() / 2

                gt_mean = mask_positions.mean(dim=1).reshape(2,1).detach()
                # TODO: use **2 instead?
                mean_loss += (pred_positions-gt_mean).abs().sum() / n_offsets

            # --- Non-instance pixels.
            #binary_mask = (instance_gt.squeeze() < 1000)
            # (id < 11) + (id == 255):
            # Do not penalize any pixel that has semantic instance
            # class (e.g. car), but have no instance information.
            binary_mask = ((instance_gt.squeeze() < 11)
                           + (instance_gt.squeeze() == 255))
            n_offsets = binary_mask.sum() * 2
            mean_loss += prediction[:,binary_mask].abs().sum() / n_offsets

        #print("mean_loss =",mean_loss)
        #print("variance_loss =",variance_loss)
        loss = (self.mean_weight * mean_loss
                + self.variance_weight * variance_loss)
        return loss


