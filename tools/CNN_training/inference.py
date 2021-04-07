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


from os import path
import re
import time
import argparse
import warnings
import math

import PIL
import h5py
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms
from apex import amp

from models import DRNSeg_inference, DRNInstance
from models.wrappers import ZeroMeanUnitVarModel, FlipAndPad
from models.DRNSeg import DRNDoubleSeg, DRNOffsetDisparity
from models.DRNDownsampled import (
        DRNDownsampledCombined, DRNDSOffsetDisparity, DRNDSDoubleSeg)
import datasets

from visualization import (save_cityscapesprediction,
        visualize_positionplusoffset, visualize_offsethsv, visualize_disparity)
from utils import check_mkdir

#import ipdb

# TODO: Remove at some point. (2019-08-01)
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

def compute_gt_disparities(disparity_gt, instance_gt):
    """
    Compute median disparity per mask and assign it to every pixel in mask.
    """
    disparities = torch.zeros(*instance_gt.size()) # h x w

    # Get all relevant instance labels (id > 1000).
    instance_ids = torch.unique(instance_gt)
    instance_ids = instance_ids[instance_ids > 1000]

    for instance_id in instance_ids:
        binary_mask = (instance_gt == instance_id)
        # Note: this median does take the one-before-middle element for even
        # length arrays. Should not matter for us.
        # See: https://github.com/torch/torch7/pull/182
        median_disparity = disparity_gt[binary_mask].float().median()

        disparities[binary_mask] = median_disparity

    return disparities

def main(model_name, model_filename, data_directory, output_directory,
         dimensions=(19,2), h5_export=True, labelImg_export=True,
         colored=False, use_instancegt=None, use_disparitygt=None,
         batch_size=1, drn_name='drn_d_22', jit=False, onnx=False):
    classification_dim = dimensions[0]
    regression_dim = dimensions[1]
    #batch_size = 1
    n_workers = 12
    # TODO: make this work, also consider downscale on input
    output_size = None # (1784, 782)

    check_mkdir(output_directory)

    mean = [0.290101, 0.328081, 0.286964]
    std = [0.182954, 0.186566, 0.184475]
    # Load model.
    if model_name == 'DRNSeg':
        print('DRNSeg.')
        dataset_kwargs = {
            'pil_transforms' : None,
            'input_transforms' : [transforms.Normalize(
                                       mean=[0.290101, 0.328081, 0.286964],
                                       std=[0.182954, 0.186566, 0.184475])] }
        regression_dim = 0
        model = DRNSeg_inference(model_name=drn_name,
                                 classes=classification_dim,
                                 pretrained_model=None)
                                 #pretrained=False)
        model.load_state_dict(torch.load('./weights/drn_d_22_cityscapes.pth'))
        model.cuda()
        model.eval()
    elif model_name == 'DRNSeg_inference':
        # actually the same as above but user defined weights
        print('DRNSeg_inference.')
        dataset_kwargs = {
            'pil_transforms' : None,
            'input_transforms' : [transforms.Normalize(
                                       mean=[0.290101, 0.328081, 0.286964],
                                       std=[0.182954, 0.186566, 0.184475])] }
        regression_dim = 0
        model = DRNSeg_inference(model_name=drn_name,
                                 classes=classification_dim,
                                 pretrained_model=None)
                                 #pretrained=False)
        model.load_state_dict(torch.load(model_filename), base=False)
        model.cuda()
        model.eval()
    elif model_name == 'DRNDownsampledCombined':
        print('DRNDownsampledCombined.')
        regression_dim = 2
        dataset_kwargs = {
                'pil_transforms' : None,
                'input_transforms' : [transforms.Normalize(
                                         mean=[0.290101, 0.328081, 0.286964],
                                         std=[0.182954, 0.186566, 0.184475])],
                'tensor_transforms' : None }
        seg_dict = torch.load(path.join(path.dirname(__file__),
                                        'weights/drn_d_22_cityscapes.pth'))
        reg_dict = torch.load(model_filename)
        model = DRNDownsampledCombined(
                       model_name=drn_name,
                       classes=classification_dim,
                       seg_dict=seg_dict,
                       reg_dict=reg_dict)
        model.cuda()
        model.eval()
    elif model_name in ('DRNDoubleSeg', 'DRNDoubleSegSL'):
        print('DRNDoubleSeg.')
        regression_dim = 2
        dataset_kwargs = {
                'pil_transforms' : None,
                'input_transforms' : [transforms.Normalize(
                                         mean=[0.290101, 0.328081, 0.286964],
                                         std=[0.182954, 0.186566, 0.184475])],
                'tensor_transforms' : None }
        model = DRNDoubleSeg(
                       model_name=drn_name,
                       classes=classification_dim)
        model.load_state_dict(torch.load(model_filename))
        model.cuda()
        model.eval()
    elif model_name in ('DRNDSDoubleSeg', 'DRNDSDoubleSegSL'):
        print('DRNDSDoubleSeg(SL).')
        regression_dim = 2
        dataset_kwargs = {
                'pil_transforms' : None,
                'input_transforms' : None,
                'tensor_transforms' : None }
        model = DRNDSDoubleSeg(
                       model_name=drn_name,
                       classes=classification_dim)
        model.load_state_dict(torch.load(model_filename))
        # Use wrapper to include normalization in onnx export
        model = ZeroMeanUnitVarModel(model, mean, std)
        model.cuda()
        model.eval()
    elif model_name == 'DRNOffsetDisparity':
        print('DRNOffsetDisparity.')
        regression_dim = 3
        dataset_kwargs = {
                'pil_transforms' : None,
                'input_transforms' : [transforms.Normalize(
                                         mean=[0.290101, 0.328081, 0.286964],
                                         std=[0.182954, 0.186566, 0.184475])],
                'tensor_transforms' : None }
        model = DRNOffsetDisparity(
                       model_name=drn_name,
                       classes=classification_dim)
        model.load_state_dict(torch.load(model_filename))
        model.cuda()
        model.eval()
    elif model_name in ('DRNDSOffsetDisparity', 'DRNDSOffsetDisparitySL',
                        'DRNDSOffsetDisparityASL'):
        print('DRNDSOffsetDisparity.')
        regression_dim = 3
        dataset_kwargs = {
                'pil_transforms' : None,
                'input_transforms' : [transforms.Normalize(
                                         mean=[0.290101, 0.328081, 0.286964],
                                         std=[0.182954, 0.186566, 0.184475])],
                'tensor_transforms' : None }
        model = DRNDSOffsetDisparity(
                       model_name=drn_name,
                       classes=classification_dim)
        model.load_state_dict(torch.load(model_filename))
        model.cuda()
        model.eval()
    elif model_name == 'DRNInstance':
        print('DRNInstance.')
        regression_dim = 2
        dataset_kwargs = {
            'pil_transforms' : None,
            'tensor_transforms' : [transforms.Normalize(
                                       mean=[0.290101, 0.328081, 0.286964],
                                       std=[0.182954, 0.186566, 0.184475])] }
        model = DRNInstance(model_name=drn_name,
                            classes=classification_dim,
                            pretrained_model=None,
                            pretrained=False)
        model.load_state_dict(torch.load('./weights/drn_d_22_cityscapes.pth'))
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

    if use_disparitygt is not None:
        if use_instancegt is None:
            use_instancegt = 'not_as_prediction'
        disp_gt_data_directory = path.join(data_directory, '..','disparities')
        disp_gt_dataset = datasets.Directory(
                disp_gt_data_directory,
                mode=None,
                suffix="disparity.png",
                pil_transforms=dataset_kwargs['pil_transforms'])
        disp_gt_dataloader = data.DataLoader(disp_gt_dataset, batch_size=batch_size,
                shuffle=False,
                num_workers=n_workers)
        dataloader = zip(dataloader, disp_gt_dataloader)

    if use_instancegt is not None:
        gt_data_directory = path.join(data_directory, '..','gtFine')
        gt_dataset = datasets.Directory(
                gt_data_directory,
                mode=None,
                suffix="instanceIds.png",
                pil_transforms=dataset_kwargs['pil_transforms'])
        gt_dataloader = data.DataLoader(gt_dataset, batch_size=batch_size,
                shuffle=False,
                num_workers=n_workers)
        dataloader = zip(dataloader, gt_dataloader)

    # apex mixed precision
    if not (jit or onnx):
        model = amp.initialize(model, opt_level="O1")

    # dry run to start up gpu
    imgs = dataset[0][1].cuda().unsqueeze(0)
    outputs = model(imgs)

    print("Start inference.")
    #upscale_factor = None
    upscale_factor = 8 if "DS" in model_name or "Downsample" in model_name\
                     else None
    if upscale_factor is not None:
        # upscale CNN semantic images only, in case upscale_factor is None
        upscale = lambda x: nn.functional.interpolate(x,
                                scale_factor=upscale_factor,
                                mode='nearest')

        #upscale = lambda x: nn.functional.interpolate(x,
        #                        scale_factor=upscale_factor,
        #                        mode='bilinear',
        #                        align_corners=True)

    total_time = 0
    for batch_data in dataloader:
        if use_disparitygt is not None:
            filenames, imgs = batch_data[0][0]
            disp_gt_filenames, gt_disps = batch_data[0][1]
            gt_filenames, gt_imgs = batch_data[1]

            re_pattern = re.compile("([a-z]*)_([0-9]*)_([0-9]*)_")
            stem = re_pattern.search(filenames[0]).groups()
            gt_stem = re_pattern.search(gt_filenames[0]).groups()
            disp_gt_stem = re_pattern.search(disp_gt_filenames[0]).groups()
            if stem != gt_stem or stem != disp_gt_stem:
                raise ValueError("Stems do not match: {} v. {} or {} v. {}"
                                 .format(stem, gt_stem, stem, disp_gt_stem))
        elif use_instancegt is not None:
            filenames, imgs = batch_data[0]
            gt_filenames, gt_imgs = batch_data[1]

            re_pattern = re.compile("([a-z]*)_([0-9]*)_([0-9]*)_")
            stem = re_pattern.search(filenames[0]).groups()
            gt_stem = re_pattern.search(gt_filenames[0]).groups()
            if stem != gt_stem:
                raise ValueError("Stems do not match: {} vs. {}."
                                 .format(stem, gt_stem))
        else:
            filenames, imgs = batch_data

        if onnx:
            # add these pre- and post-processing steps to the model
            flip_and_pad = FlipAndPad(model)
            torch.onnx.export(flip_and_pad, imgs.cuda().detach(),
                              'tmp/{}_zmuv_fp.onnx'
                              .format(path.basename(model_filename)),
                              input_names=["input.1"],
                              output_names=["output.1"],
                              verbose=False,
                              opset_version=10)
            onnx = False

        if jit:
            with torch.no_grad():
                sm = torch.jit.trace(model, imgs.cuda().detach())
                sm.save("tmp/traced_{}.pt"
                        .format(path.basename(model_filename)))
                del sm
            jit = False

        start_time = time.time()
        imgs = imgs.cuda()
        torch.cuda.synchronize()
        end_time = time.time()
        print("Copy time (est.): {}".format(end_time-start_time))
        with torch.no_grad():
            outputs = model(imgs)
        outputs = outputs.detach()
        output_filenames = [f.replace(data_directory, output_directory)\
                            for f in filenames]
        for output_filename in output_filenames:
            check_mkdir(path.dirname(output_filename))

        # Remove padding if necessary.
        if dataset.padding_size is not None:
            left, top, right, bottom = dataset.padding_size
            right = imgs.size(3) - right
            bottom = imgs.size(2) - bottom
            outputs = outputs[:,:,top:bottom,left:right]

        # Upscale if neccessary.
        #~ if upscale_factor is not None and regression_dim != 0:
        #~     #outputs[:,-regression_dim:] *= upscale_factor
        #~     # Fixed to "-2" since disparity does not need rescaling.
        #~     outputs[:,-2:] *= upscale_factor
        #~     outputs = upscale(outputs)

        if classification_dim != 0:
            outputs[:,:classification_dim] = -outputs[:,:classification_dim]
            classification_output = outputs[:,:classification_dim]
            if upscale is not None:
                predictions = upscale(classification_output)
            predictions = predictions.argmin(dim=1)
            for filename, prediction in zip(output_filenames, predictions):
                save_cityscapesprediction(prediction, filename, colored,
                                          resize=output_size)
        if regression_dim != 0:
            regression_output = outputs[:,-regression_dim:]
        #else:
        #    # TODO: Why not actually?
        #    raise ValueError("Code can't handle no regression output anymore.")

        if use_instancegt is not None:
            gt_offsets = compute_instance_offsets(gt_imgs[0,0])
            #~ if upscale_factor is not None and "Downsampled" not in model_name:
            #~     gt_offsets = upscale(gt_offsets.unsqueeze(0)*upscale_factor)
            #~     gt_offsets = gt_offsets.squeeze()
            if use_instancegt == 'as_prediction':
                print("Using ground truth offsets.")
                outputs[:,-2:] = gt_offsets
                regression_output = gt_offsets
            visualize_positionplusoffset(
                    regression_output,
                    gt_filenames[0].replace(gt_data_directory, output_directory)[:-4],
                    gt_offsets)

        if regression_dim != 0:
            # TODO: add batch loop
            filename_hsv = filenames[0].replace(data_directory, output_directory)
            if regression_dim > 2:
                filename_disparity = filename_hsv[:-4] + "_disparity"
                visualize_disparity(
                        regression_output[0,0,:,:],
                        filename_disparity)
            filename_hsv = filename_hsv[:-4] + "_hsv"
            visualize_offsethsv(
                    regression_output,
                    filename_hsv)

        if use_disparitygt is not None:
            # Match instance gt mask and disparities, compute mean disp.
            gt_disparities = compute_gt_disparities(gt_disps[0,0],gt_imgs[0,0])
            gt_disparities /= 255
            #if upscale_factor is not None and "Downsampled" not in model_name:
            #    gt_offsets = upscale(gt_offsets.unsqueeze(0)*upscale_factor)
            #    gt_offsets = gt_offsets.squeeze()
            #if use_disparitygt == 'as_prediction':
            print("Using ground truth disparity medians.")
            gt_disparities = gt_disparities.unsqueeze(0).unsqueeze(0)
            outputs = torch.cat((outputs[:,:classification_dim].cpu(),
                                 gt_disparities,
                                 outputs[:,-2:].cpu()), dim=1)
            #print("outputs.size()",outputs.size())
            # TODO: visualize?

        if h5_export:
            with torch.no_grad():
                # start timer
                torch.cuda.synchronize()
                start_time = time.time()

                fap = FlipAndPad(model)
                output = fap(imgs)[0]

                # end timer
                torch.cuda.synchronize()
                end_time = time.time()
                total_time += end_time-start_time
            print("Inference time (est.): {}".format(end_time-start_time))
            outputs = (output,)

            for filename, output in zip(output_filenames, outputs):
                output = output.cpu()

                filename = filename.replace("_leftImg8bit", "")
                filename = filename.replace(".png", "_probs.h5")
                filename = path.abspath(filename)
                print("Prediction saved in {}.".format(filename))
                h5_file = h5py.File(filename, 'w')
                h5_file.create_dataset('nlogprobs', data=output)

    time_per_batch = total_time / len(dataloader)
    fps = 1./time_per_batch
    print("Time per batch {} s, {} fps.".format(time_per_batch, fps))
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
    parser.add_argument("--disparitygt", type=str, default=None,
                        help="Output 'ground truth' median disparity per "
                             "instance mask.")
    parser.add_argument("--save-dir", type=str, default="tmp/inference/",
                        help="Path to output directory.")
    parser.add_argument("--model", type=str,
                        help="Choose a model.")
    parser.add_argument("--batch-size", "-b", type=int, default=1,
                        help="Set batch size.")
    parser.add_argument("--drn-name", type=str, default='drn_d_22',
                        help="DRN base model. Default: drn_d_22.")
    parser.add_argument("--jit", action="store_true",
                        help="JIT the model to import it in CPP.")
    parser.add_argument("--onnx", action="store_true",
                        help="Export the model in onnx format.")
    args = parser.parse_args()

    main(args.model, args.WEIGHTS, args.IMG_PATH, args.save_dir,
         h5_export=not args.no_h5, labelImg_export=not args.no_labelImg,
         colored=args.colored, use_instancegt=args.instancegt,
         use_disparitygt=args.disparitygt, batch_size=args.batch_size,
         drn_name=args.drn_name, jit=args.jit,
         onnx=args.onnx)
