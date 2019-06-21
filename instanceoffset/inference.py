#!/usr/bin/env python

# This file is part of Instance Stixels.
# Original:
# Copyright (c) 2017, Fisher Yu
# BSD 3-Clause License
# See: https://github.com/fyu/drn

# Modifications:
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
# along with Instance Stixels.  If not, see <https://www.gnu.org/licenses/>.


import os
import re
import time
import argparse
import warnings

import PIL
import h5py
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms

from models.DRNDownsampled import DRNDownsampledCombined
import datasets

from visualization import save_cityscapesprediction,\
    visualize_positionplusoffset, visualize_offsethsv
from utils import check_mkdir

def compute_instance_offsets(instance_gt):
    """
    Compute offset per pixel to center of mass of corresponding instance
    mask.
    """
    offsets = torch.zeros((2,*instance_gt.size())) # 1 x h x w

    # Get all relevant instance labels (id > 1000).
    instance_ids = torch.unique(instance_gt)
    instance_ids = instance_ids[instance_ids > 1000]
    # Compute center of mass of each instance mask and per pixel offsets.
    for instance_id in instance_ids:
        binary_mask = (instance_gt == instance_id)
        instance_positions = torch.nonzero(binary_mask).float().t()

        center_of_mass = instance_positions.mean(dim=1).reshape(2,1)
        offsets[:,binary_mask] = center_of_mass - instance_positions

    return offsets

def main(model_name, model_filename, data_directory, output_directory,
         dimensions=(19,2), h5_export=True, labelImg_export=True,
         colored=False, use_instancegt=None):
    classification_dim = dimensions[0] 
    regression_dim = dimensions[1]
    batch_size = 1
    n_workers = 12

    check_mkdir(output_directory)

    # Load model.
    if model_name == 'DRNDownsampledCombined':
        print('DRNDownsampledCombined.')
        regression_dim = 2
        dataset_kwargs = {
                'pil_transforms' : None,
                'input_transforms' : [transforms.Normalize(
                                         mean=[0.290101, 0.328081, 0.286964],
                                         std=[0.182954, 0.186566, 0.184475])],
                'tensor_transforms' : None }
        seg_dict = torch.load(os.path.join(
                        os.path.dirname(__file__),
                        'weights/drn_d_22_cityscapes.pth'))
        reg_dict = torch.load(model_filename)
        model = DRNDownsampledCombined(
                       model_name='drn_d_22',
                       classes=classification_dim,
                       seg_dict=seg_dict,
                       reg_dict=reg_dict)
        model.cuda()
        model.eval()
    else:
        raise ValueError("Model \"{}\" not found!".format(model_name))

    # Create data loader.
    print("Creating data loader.")
    dataset = datasets.Directory(data_directory,
                                 suffix="leftImg8bit.png",
                                 **dataset_kwargs)
    dataloader = data.DataLoader(dataset, batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=n_workers)

    if use_instancegt is not None:
        gt_data_directory = os.path.join(data_directory, '..','gtFine')
        gt_dataset = datasets.Directory(
                gt_data_directory,
                mode=None,
                suffix="instanceIds.png",
                pil_transforms=dataset_kwargs['pil_transforms'])
        gt_dataloader = data.DataLoader(gt_dataset, batch_size=batch_size,
                shuffle=False,
                num_workers=n_workers)
        dataloader = zip(dataloader, gt_dataloader)

    print("Start inference.")
    upscale_factor = 8 # None # 2
    if upscale_factor is not None:
        upscale = lambda x: nn.functional.interpolate(x,
                                scale_factor=upscale_factor,
                                mode='nearest')

    for batch_data in dataloader:
        if use_instancegt is None:
            filenames, imgs = batch_data
        else:
            filenames, imgs = batch_data[0]
            gt_filenames, gt_imgs = batch_data[1]

            re_pattern = re.compile("([a-z]*)_([0-9]*)_([0-9]*)_")
            stem = re_pattern.search(filenames[0]).groups()
            gt_stem = re_pattern.search(gt_filenames[0]).groups()
            if stem != gt_stem:
                raise ValueError("Stems do not match: {} vs. {}."
                                 .format(stem, gt_stem))

        start_time = time.time()
        outputs = model(imgs.cuda()).detach()
        end_time = time.time()
        print("Inference time (est.): {}".format(end_time-start_time))
        output_filenames = [f.replace(data_directory, output_directory)\
                            for f in filenames]

        # Remove padding if necessary.
        if dataset.padding_size is not None:
            left, top, right, bottom = dataset.padding_size
            right = imgs.size(3) - right
            bottom = imgs.size(2) - bottom
            outputs = outputs[:,:,top:bottom,left:right]

        # Upscale if neccessary.
        if upscale_factor is not None and regression_dim != 0:
            outputs[:,-regression_dim:] *= upscale_factor
            outputs = upscale(outputs)

        if classification_dim != 0:
            outputs[:,:classification_dim] = -outputs[:,:classification_dim]
            classification_output = outputs[:,:classification_dim]
            predictions = classification_output.argmin(dim=1)
            for filename, prediction in zip(output_filenames, predictions):
                save_cityscapesprediction(prediction, filename, colored,
                                          resize=None)
        if regression_dim != 0:
            regression_output = outputs[:,-regression_dim:]
        else:
            raise ValueError("Code can't handle no regression output anymore.")

        if use_instancegt is not None:
            gt_offsets = compute_instance_offsets(gt_imgs[0,0])
            if upscale_factor is not None and "Downsampled" not in model_name:
                gt_offsets = upscale(gt_offsets.unsqueeze(0)*upscale_factor)
                gt_offsets = gt_offsets.squeeze()
            if use_instancegt == 'as_prediction':
                print("Using ground truth offsets.")
                outputs[:,-2:] = gt_offsets
                regression_output = gt_offsets
            visualize_positionplusoffset(
                    regression_output,
                    gt_filenames[0].replace(gt_data_directory, output_directory)[:-4],
                    gt_offsets)
        filename_hsv = filenames[0].replace(data_directory, output_directory)
        filename_hsv = filename_hsv[:-4] + "_hsv"
        visualize_offsethsv(
                regression_output,
                filename_hsv)

        if h5_export:
            for filename, output in zip(output_filenames, outputs):
                output = output.cpu()
                filename = filename.replace("_leftImg8bit", "")
                filename = filename.replace(".png", "_probs.h5")
                h5_file = h5py.File(filename, 'w')
                h5_file.create_dataset('nlogprobs',
                                       data=output.squeeze().permute(1,2,0))
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description="Run inference of a trained NN.")
    parser.add_argument("WEIGHTS", type=str,
                        help="A .pth file where the weights of the trained "
                             "model are stored.")
    parser.add_argument("IMG_PATH", type=str,
                        help="Path to the RGB input image files.")
    parser.add_argument("--no-labelImg", action="store_true",
                        help="Do not save labelImgs.")
    parser.add_argument("--no-h5", action="store_true",
                        help="Do not save negative log probabilites h5 files.")
    parser.add_argument("--colored", action="store_true",
                        help="Save a colorful version of the prediction.")
    parser.add_argument("--instancegt", type=str, default=None,
                        help="Output ground truth instance offsets instead of "
                             "CNN output.")
    parser.add_argument("--save-dir", type=str, default="tmp/inference/",
                        help="Path to output directory.")
    parser.add_argument("--model", type=str,
                        help="Choose a model.")
    args = parser.parse_args()

    main(args.model, args.WEIGHTS, args.IMG_PATH, args.save_dir, 
         h5_export=not args.no_h5, labelImg_export=not args.no_labelImg,
         colored=args.colored, use_instancegt=args.instancegt)
