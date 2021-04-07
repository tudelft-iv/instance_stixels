# This file is part of Instance Stixels:
# https://github.com/tudelft-iv/instance-stixels
#
# Original:
# Copyright (c) 2017, Fisher Yu
# BSD 3-Clause License
#
# Modifications:
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

from os import path
import argparse
import subprocess
import json
import time

import numpy as np

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from apex import amp

# TODO: move visualization stuff to visualization module
from matplotlib import pyplot as plt
import PIL.Image

from datasets import Cityscapes
from datasets.transforms import ModeDownsample, MultiImgRandomHorizontalFlip

from models import DRNSeg
from models.DRNDownsampled import (
        DRNRegressionDownsampled, DRNDSOffsetDisparity, DRNDSDoubleSeg)
from models.DRNSeg import (
        DRNDoubleSeg, DRNOffsetDisparity, DRNMultifield, DRNMultifieldMax,
        DRNRegressionOnly, DRNMultifieldAfterUp)
from models import DRNSeg_inference, DRNInstance
import losses
from visualization import (
        visualize_positionplusoffset, visualize_semantics, visualize_offsethsv,
        save_cityscapesprediction, visualize_disparity)
from utils import check_mkdir, print_bytes, parse_semantic_evaluation_output

#import ipdb

class ModelWrapper:
    def __init__(self, model_name, n_classes, weights=None,
                 drn_name='drn_d_22'):
        print("--- Load model.")
        self.drn_name = drn_name
        val_dataset_kwargs = None
        regression_channels = 2
        types = None
        self.downsample_pil = None
        if model_name == 'DRNSeg_inference':
            regression_channels = 0
            types = ('left','semantic_gt')
            regression_loss = None # nn.MSELoss(reduction='mean')
            classification_loss = nn.NLLLoss(reduction='mean',
                                             ignore_index=255)
            train_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'left' : transforms.ColorJitter(
                                        brightness=0.2,
                                        contrast=0.2,
                                        saturation=0.1,
                                        hue=0.1),
                            'combined' : MultiImgRandomHorizontalFlip()#,
                            } }}
            val_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                            }}
            model = DRNSeg_inference(
                                model_name=drn_name,
                                classes=n_classes,
                                pretrained_model=None,
                                pretrained=True)
            #model.load_state_dict(torch.load('./weights/drn_d_22_cityscapes.pth'))
            model.cuda()
            parameters = model.base.optim_parameters()
        elif model_name == 'DRNRegressionDownsampled':
            classification_loss = None
            regression_loss = nn.MSELoss(reduction='mean')
            dataset_kwargs = {
                    'pil_transforms' : None,
                    'gt_pil_transforms' : ModeDownsample(8),
                    #'fit_gt_pil_transforms' : [transforms.Resize(
                    #                                size=(784//8, 1792//8),
                    #                                interpolation=2)],
                    'input_transforms' : [transforms.Normalize(
                                             mean=[0.290101, 0.328081, 0.286964],
                                             std=[0.182954, 0.186566, 0.184475])],
                    'tensor_transforms' : None }
            model = DRNRegressionDownsampled(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained_dict=torch.load('./weights/drn_d_22_cityscapes.pth'))
            model.cuda()
            parameters = model.parameters()
        elif model_name == 'DRNDoubleSegSL':
            print("Training downsampled double seg model with separate loss.")
            regression_loss = losses.OffsetLossSL(**weights)
            #regression_loss = nn.MSELoss(reduction='mean')
            classification_loss = nn.NLLLoss(reduction='mean',
                                             ignore_index=255)
            types = ('left','semantic_gt','instance_gt_raw')
            train_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'left' : transforms.ColorJitter(
                                        brightness=0.2,
                                        contrast=0.2,
                                        saturation=0.1,
                                        hue=0.1),
                            'combined' : MultiImgRandomHorizontalFlip()#,
                            } }}
            val_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                            }}
            model = DRNDoubleSeg(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained=True)
                           #pretrained_dict=torch.load('./weights/drn_d_22_cityscapes.pth'))
            model.cuda()
            #parameters = model.optim_seg_parameters()
            parameters = model.optim_parameters()
        elif model_name == 'DRNDoubleSeg':
            print("Training double seg model.")
            regression_loss = nn.SmoothL1Loss(reduction='mean')
            #regression_loss = nn.MSELoss(reduction='mean')
            classification_loss = nn.NLLLoss(reduction='mean',
                                             ignore_index=255)
            types = ('left','semantic_gt','instance_gt')
            train_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'left' : transforms.ColorJitter(
                                        brightness=0.2,
                                        contrast=0.2,
                                        saturation=0.1,
                                        hue=0.1),
                            'combined' : MultiImgRandomHorizontalFlip()#,
                            } }}
            val_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                            }}
            model = DRNDoubleSeg(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained=True)
                           #pretrained_dict=torch.load('./weights/drn_d_22_cityscapes.pth'))
            model.cuda()
            #parameters = model.optim_seg_parameters()
            parameters = model.optim_parameters()
        elif model_name == 'DRNOffsetDisparity':
            print("Training offset disparity model.")
            regression_channels = 3
            types = ('left','semantic_gt','instance_gt','disparity')
            regression_loss = nn.SmoothL1Loss(reduction='mean')
            #regression_loss = nn.MSELoss(reduction='mean')
            classification_loss = nn.NLLLoss(reduction='mean',
                                             ignore_index=255)
            train_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'combined' : MultiImgRandomHorizontalFlip()#,
                            } }}
            val_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                            }}
            model = DRNOffsetDisparity(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained=True)
            model.cuda()
            #parameters = model.optim_seg_parameters()
            parameters = model.optim_parameters()
        elif model_name == 'DRNDSDoubleSeg':
            print("Training downsampled double seg model.")
            regression_channels = 2
            types = ('left','semantic_gt','instance_gt')
            #regression_loss = losses.OffsetLossSL(**weights)
            regression_loss = nn.SmoothL1Loss(reduction='mean')
            classification_loss = nn.NLLLoss(reduction='mean',
                                             ignore_index=255)
            self.downsample_pil = transforms.Resize(size=(784//8, 1792//8),
                                                    interpolation=2)
            train_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'instance_gt' : ModeDownsample(8),
                            'semantic_gt' : ModeDownsample(8),
                            'combined' : MultiImgRandomHorizontalFlip()#,
                            } }}
            val_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'instance_gt' : ModeDownsample(8),
                            'semantic_gt' : ModeDownsample(8)
                            }
                        }}
            model = DRNDSDoubleSeg(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained=True)
            model.cuda()
            #parameters = model.optim_seg_parameters()
            parameters = model.optim_parameters()
        elif model_name == 'DRNDSDoubleSegSL':
            print("Training downsampled double seg model with separated loss.")
            regression_channels = 2
            types = ('left','semantic_gt','instance_gt_raw')
            regression_loss = losses.OffsetLossSL(**weights)
            #regression_loss = nn.MSELoss(reduction='mean')
            classification_loss = nn.NLLLoss(reduction='mean',
                                             ignore_index=255)
            self.downsample_pil = transforms.Resize(size=(784//8, 1792//8),
                                                    interpolation=2)
            train_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'instance_gt_raw' : ModeDownsample(8),
                            'semantic_gt' : ModeDownsample(8),
                            'combined' : MultiImgRandomHorizontalFlip()#,
                            } }}
            val_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'instance_gt_raw' : ModeDownsample(8),
                            'semantic_gt' : ModeDownsample(8)
                            }
                        }}
            model = DRNDSDoubleSeg(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained=True)
            model.cuda()
            #parameters = model.optim_seg_parameters()
            parameters = model.optim_parameters()
        elif model_name in ('DRNDSOffsetDisparitySL', 'DRNDSOffsetDisparityASL'):
            print("Training downsampled offset disparity model with separated "
                  "loss.")
            regression_channels = 3
            types = ('left','semantic_gt','instance_gt_raw','disparity_raw')
            abs_variance = model_name[-3:] == 'ASL'
            regression_loss = losses.DisparityOffsetLossSL(
                                **weights, abs_variance=abs_variance)
            #regression_loss = nn.MSELoss(reduction='mean')
            classification_loss = nn.NLLLoss(reduction='mean',
                                             ignore_index=255)
            self.downsample_pil = transforms.Resize(size=(784//8, 1792//8),
                                                    interpolation=2)
            train_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'instance_gt_raw' : ModeDownsample(8),
                            'disparity_raw' : ModeDownsample(8),
                            'semantic_gt' : ModeDownsample(8),
                            'combined' : MultiImgRandomHorizontalFlip()#,
                            } }}
            val_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'instance_gt_raw' : ModeDownsample(8),
                            'disparity_raw' : ModeDownsample(8),
                            'semantic_gt' : ModeDownsample(8)
                            }
                        }}
            model = DRNDSOffsetDisparity(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained=True)
            model.cuda()
            #parameters = model.optim_seg_parameters()
            parameters = model.optim_parameters()
        elif model_name == 'DRNDSOffsetDisparity':
            print("Training downsampled offset disparity model.")
            regression_channels = 3
            types = ('left','semantic_gt','instance_gt','disparity')
            regression_loss = nn.SmoothL1Loss(reduction='mean')
            #regression_loss = nn.MSELoss(reduction='mean')
            classification_loss = nn.NLLLoss(reduction='mean',
                                             ignore_index=255)
            self.downsample_pil = transforms.Resize(size=(784//8, 1792//8),
                                                    interpolation=2)
            train_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'instance_gt' : ModeDownsample(8),
                            'disparity' : ModeDownsample(8),
                            'semantic_gt' : ModeDownsample(8),
                            'combined' : MultiImgRandomHorizontalFlip()#,
                            } }}
            val_dataset_kwargs = {
                    'transforms' : {
                        'tensor' : {
                            'pre' : transforms.ToTensor(),
                            'left': transforms.Normalize(
                                        mean=[0.290101, 0.328081, 0.286964],
                                        std=[0.182954, 0.186566, 0.184475])},
                        'pil' : {
                            'instance_gt' : ModeDownsample(8),
                            'disparity' : ModeDownsample(8),
                            'semantic_gt' : ModeDownsample(8)
                            }
                        }}
            model = DRNDSOffsetDisparity(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained=True)
            model.cuda()
            #parameters = model.optim_seg_parameters()
            parameters = model.optim_parameters()
        elif model_name == 'DRNRegressionOnly':
            #regression_loss = nn.SmoothL1Loss(reduction='sum')
            regression_loss = nn.MSELoss(reduction='sum')
            classification_loss = None
            dataset_kwargs = {
                    'pil_transforms' : None,
                    'input_transforms' : [transforms.Normalize(
                                             mean=[0.290101, 0.328081, 0.286964],
                                             std=[0.182954, 0.186566, 0.184475])],
                    'tensor_transforms' : None }
            model = DRNRegressionOnly(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained_dict=torch.load('./weights/drn_d_22_cityscapes.pth'))
            model.cuda()
            #parameters = model.optim_seg_parameters()
            parameters = model.parameters()
        elif model_name == 'DRNMultifield':
            print("Multifield.")
            regression_loss = nn.MSELoss(reduction='sum')
            classification_loss = nn.NLLLoss(reduction='sum',
                                             ignore_index=255)
            dataset_kwargs = {
                    'pil_transforms' : None,
                    'input_transforms' : [transforms.Normalize(
                                             mean=[0.290101, 0.328081, 0.286964],
                                             std=[0.182954, 0.186566, 0.184475])],
                    'tensor_transforms' : None }
            model = DRNMultifield(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained_dict=torch.load('./weights/drn_d_22_cityscapes.pth'))
            model.cuda()
            #parameters = model.optim_seg_parameters()
            #parameters = model.optim_parameters()
            print("Optimizing field parameters only!!")
            parameters = model.optim_field_parameters()
        elif model_name == 'DRNMultifieldAfterUp':
            print("MultifieldAfterUp.")
            regression_loss = nn.MSELoss(reduction='sum')
            classification_loss = nn.NLLLoss(reduction='sum',
                                             ignore_index=255)
            dataset_kwargs = {
                    'pil_transforms' : None,
                    'input_transforms' : [transforms.Normalize(
                                             mean=[0.290101, 0.328081, 0.286964],
                                             std=[0.182954, 0.186566, 0.184475])],
                    'tensor_transforms' : None }
            model = DRNMultifield(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained_dict=torch.load('./weights/drn_d_22_cityscapes.pth'))
            model.cuda()
            #parameters = model.optim_seg_parameters()
            parameters = model.optim_parameters()
        elif model_name == 'DRNMultifieldMax':
            print(model_name)
            regression_loss = nn.MSELoss(reduction='sum')
            classification_loss = nn.NLLLoss(reduction='sum',
                                             ignore_index=255)
            dataset_kwargs = {
                    'pil_transforms' : None,
                    'input_transforms' : [transforms.Normalize(
                                             mean=[0.290101, 0.328081, 0.286964],
                                             std=[0.182954, 0.186566, 0.184475])],
                    'tensor_transforms' : None }
            model = DRNMultifieldMax(
                           model_name=drn_name,
                           classes=n_classes,
                           pretrained_dict=torch.load('./weights/drn_d_22_cityscapes.pth'))
            model.cuda()
            #parameters = model.optim_seg_parameters()
            parameters = model.optim_parameters()
        elif model_name == 'DRNInstance':
            regression_loss = nn.MSELoss(reduction='mean')
            classification_loss = nn.NLLLoss(reduction='mean',
                                             ignore_index=255)
            dataset_kwargs = {
                    'pil_transforms' : None,
                    'input_transforms' : [transforms.Normalize(
                                             mean=[0.290101, 0.328081, 0.286964],
                                             std=[0.182954, 0.186566, 0.184475])],
                    'tensor_transforms' : None }
            model = DRNInstance(regression_head=DRNSeg.RegressionHead_3layers,
                                #train_all=False,
                                model_name=drn_name,
                                classes=n_classes,
                                pretrained_model=None,
                                pretrained=False)
            model.load_state_dict(torch.load('./weights/drn_d_22_cityscapes.pth'))
            model.cuda()
            #parameters = model.regression_parameters()
            parameters = model.parameters()
        elif model_name == 'DRNInstance_noclass':
            classification_loss = None
            dataset_kwargs = {
                    'pil_transforms' : None,
                    'input_transforms' : [transforms.Normalize(
                                             mean=[0.290101, 0.328081, 0.286964],
                                             std=[0.182954, 0.186566, 0.184475])],
                    'tensor_transforms' : None }
            model = DRNInstance(regression_head=DRNSeg.RegressionHead_3layers,
                                train_all=False,
                                model_name=drn_name,
                                classes=n_classes,
                                pretrained_model=None,
                                pretrained=False)
            model.load_state_dict(torch.load('./weights/drn_d_22_cityscapes.pth'))
            model.cuda()
            parameters = model.regression_parameters()
        else:
            raise ValueError("Model \"{}\" not found!".format(model_name))

        self.NN = model
        self.model_name = model_name
        self.n_classes = n_classes
        self.regression_channels = regression_channels
        self.regression_loss = regression_loss
        self.classification_loss = classification_loss
        self.parameters = parameters
        self.types = types if types is not None else\
                     ('left','semantic_gt','instance_gt')
        if val_dataset_kwargs is not None:
            self.dataset_kwargs = train_dataset_kwargs
            self.val_dataset_kwargs = val_dataset_kwargs
        else:
            self.dataset_kwargs = train_dataset_kwargs
            self.val_dataset_kwargs = train_dataset_kwargs

    def compute_loss(self, data_loader, separate=False):
        closs = 0
        rloss = 0
        for batch_data in data_loader:
            with torch.no_grad():
                outputs = self.NN(batch_data[0].cuda(non_blocking=True))
                batch_losses = self.batch_loss(batch_data, outputs, separate)
                closs += batch_losses[0]
                rloss += batch_losses[1]
        closs /= len(data_loader)
        rloss /= len(data_loader)
        closs = float(closs)
        if len(rloss.size()) > 0:
            rloss = tuple(float(l) for l in rloss)
            return (closs, *rloss)

        rloss = float(rloss)
        return (closs, rloss)

    def save_model(self, output_directory, suffix=""):
        model_file = path.join(output_directory,
                               "{}_{}.pth".format(self.model_name, suffix))
        torch.save(self.NN.state_dict(), model_file)
        return path.abspath(model_file)

    def validation_snapshot(self, model_file, output_directory,
                            cityscapes_root, batch_size=1,
                            val_split='subtrainval'):
        torch.cuda.empty_cache()
        directories = {
                'img_output' : path.abspath(path.join(output_directory,
                                                      'pred')),
                'img_input'  : path.abspath(path.join(cityscapes_root,
                                                      'leftImg8bit',
                                                      val_split)),
                'img_gt'     : path.abspath(path.join(cityscapes_root,
                                                      'gtFine',
                                                      val_split,
                                                      '*/*labelIds.png')) }
        directories['img_pred'] =\
                path.abspath(path.join(directories['img_output'],
                                       '*/*labelImg.png'))

        # adapted from run.py
        #model_file = "weights/Net_DRNRegDs_epoch45.pth"
        model = self.model_name
        command = ["/bin/bash", "-i", # interactive shell for conda
                   "run_segmentation.sh",
                   model_file,
                   directories['img_input'],
                   directories['img_output'],
                   model,
                   "--no-h5",
                   "--drn-name", self.drn_name,
                   "-b {}".format(batch_size)] #+ options
        print("Running segmentation: "+" ".join(command))
        subprocess.check_output(command)
        #subprocess.check_call(command) # for debugging

        # adapted from run.py
        command = ["/bin/bash", "-i", # interactive shell for conda
                   "../evaluation/run_semanticevaluation.sh",
                   # The string "pred" has to be in the path for the eval
                   # script.
                   directories['img_pred'],
                   directories['img_gt']]
        print("Running semantic CNN evaluation: "+" ".join(command))
        output = subprocess.check_output(command)
        print_bytes(output)
        semantic_score = parse_semantic_evaluation_output(output)

        return semantic_score

    def validation_visual(self, validation_idxs, output_directory, epoch=0):
        # visualize validation imgs.
        check_mkdir('{}/offsets'.format(output_directory))
        check_mkdir('{}/disparity'.format(output_directory))
        check_mkdir(output_directory)
        check_mkdir('{}/offsets/means'.format(output_directory))
        check_mkdir('{}/semantics'.format(output_directory))
        check_mkdir('{}/semantics/overlay'.format(output_directory))
        self.NN.eval()
        for validation_idx in validation_idxs:
            img_pil = self.val_set.get_images(validation_idx)['left']
            batch_data = self.val_set[validation_idx][:3]
            if len(batch_data) > 2:
                img, _, offset_gt = batch_data
                if offset_gt.size(0) == 1:
                    offset_gt = None
            else:
                img, _ = batch_data
                offset_gt = None
            img = img.unsqueeze(0).cuda()
            with torch.no_grad():
                outputs = self.NN(img)
            #outputs = upscale(outputs)
            #offset_gt = upscale(offset_gt.unsqueeze(0)).squeeze()
            if self.downsample_pil is not None:
                img_pil = self.downsample_pil(img_pil)
            epoch_filename = 'id{:03}_epoch{:05}'\
                             .format(validation_idx, epoch)
            if self.classification_loss is not None:
                visualize_semantics(
                        img_pil, outputs[:,:self.n_classes,:,:],
                        "{}/semantics/{}"
                        .format(output_directory, epoch_filename),
                        "{}/semantics/overlay/{}"
                        .format(output_directory, epoch_filename))
            if self.regression_loss is not None:
                if self.regression_channels > 2:
                    visualize_disparity(
                        outputs[:,-self.regression_channels,:,:].detach(),
                        "{}/disparity/{}"
                        .format(output_directory, epoch_filename))
                visualize_offsethsv(
                        outputs[:,-self.regression_channels:,:,:].detach(),
                        "{}/offsets/{}"
                        .format(output_directory, epoch_filename))
                if offset_gt is not None:
                    visualize_positionplusoffset(
                            outputs[:,-self.regression_channels:,:,:].detach(),
                            "{}/offsets/means/{}"
                            .format(output_directory, epoch_filename),
                            groundtruth=offset_gt)
                # TODO: remove!
                if self.model_name.startswith('DRNMultifield'):
                    with torch.no_grad():
                        _, y_fields, x_fields, activations =\
                                model(img, all_fields=True)

                    # TODO: visualize activation
                    activations = activations.detach().cpu().numpy()
                    #np.moveaxis(activation, 0, -1)
                    for i, (y_field, x_field, activation) in\
                            enumerate(zip(y_fields[0], x_fields[0],
                                          activations[0])):
                        base_folder = '{}/offsets/{}'\
                                      .format(output_directory, i)
                        check_mkdir('{}/activation'
                                    .format(base_folder))
                        check_mkdir('{}/offsets'
                                    .format(base_folder))
                        check_mkdir('{}/means'
                                    .format(base_folder))
                        yx_field = torch.cat((y_field.unsqueeze(0),
                                              x_field.unsqueeze(0)),
                                             dim=0)
                        yx_field = yx_field.unsqueeze(0)
                        visualize_offsethsv(
                                yx_field,
                                "{}/offsets/{}"
                                .format(base_folder, epoch_filename))
                        visualize_positionplusoffset(
                                yx_field,
                                "{}/means/{}"
                                .format(base_folder, epoch_filename),
                                groundtruth=offset_gt)

                        activation = np.uint8(activation * 255)
                        activation_pil = PIL.Image.fromarray(activation)
                        activation_pil.save('{}/activation/{}.png'
                                            .format(base_folder,
                                                    epoch_filename))

    def batch_loss(self, batch_data, outputs=None, separate=False):
        img = batch_data[0].cuda(non_blocking=True)
        semantic_gt = batch_data[1].detach().cuda(non_blocking=True)
        if len(batch_data) > 2:
            instance_offset_gt = batch_data[2].detach().cuda(non_blocking=True)
        if len(batch_data) > 3:
            disparity_gt = batch_data[3].detach().cuda(non_blocking=True)

        if outputs is None:
            outputs = self.NN(img)

        closs = 0
        rloss = 0
        if self.regression_loss is not None:
            nonzero = False
            if nonzero:
                non_zero_mask = instance_offset_gt != 0
                predicted_offset = outputs[:,-self.regression_channels:]
                predicted_offset = predicted_offset[non_zero_mask]
                instance_offset_gt = instance_offset_gt[non_zero_mask]
            else:
                predicted_offset = outputs[:,-self.regression_channels:]

            if "SL" in self.model_name:
                rloss = self.regression_loss(predicted_offset,
                                             instance_offset_gt,
                                             disparity_gt, separate)
            else:
                rloss = self.regression_loss(predicted_offset,
                                             instance_offset_gt)

        if self.classification_loss is not None:
            closs = self.classification_loss(outputs[:,:self.n_classes],
                                             semantic_gt)
        return closs, rloss


def plot_losses(metrics, filename):
    try:
        rlosses = metrics['regression']
        closses = metrics['classification']
        epochs_measured = metrics['epochs']
    except KeyError:
        return

    if 'semantic' in metrics.keys():
        plt.plot(epochs_measured, metrics['semantic'])
        plt.savefig('{}_semantic.svg'.format(filename))
        plt.close('all')

    for key in ('offset_mean_loss', 'offset_variance_loss',
                'disparity_mean_loss', 'disparity_variance_loss'):
        if key in metrics.keys():
            plt.plot(epochs_measured, metrics[key])
            plt.savefig('{}_{}.svg'.format(filename, key))
            plt.close('all')

    plt.plot(epochs_measured, rlosses)
    plt.savefig('{}_rlosses.svg'.format(filename))
    plt.close('all')
    plt.plot(epochs_measured, closses)
    plt.savefig('{}_closses.svg'.format(filename))
    plt.close('all')
    plt.plot(epochs_measured, np.add(rlosses,closses))
    plt.savefig('{}_losses.svg'.format(filename))
    plt.close('all')


def main(model_name, initial_validation):
    # TODO: parse args.
    # --- Tunables.
    # 32GB DRNDSOffsetDisparity, cropped -> 18
    # 12GB DRNDSOffsetDisparity, cropped -> 6
    # 12GB DRNOffsetDisparity, cropped -> 4
    # 12GB DRNOffsetDisparity, original -> 3
    # 12GB DRNDSOffsetDisparity, original -> not supported yet:
    # resize is based on resolution 1792x784
    batch_size = 6 # 6
    n_workers = 21
    n_semantic_pretrain = 0 # 500 # First train only on semantics.
    n_epochs = 500
    validation_step = 5
    train_split = 'subtrain' # 'train'
    val_split = 'subtrainval' # 'val'
    validate_on_train = False # Note: this doesn't include semantic performance.
    train_set_length = 24 # 24 # None
    #cityscapes_directory = "/home/thehn/cityscapes/original"
    cityscapes_directory = "/home/thehn/cityscapes/cropped_cityscapes"
    #cityscapes_directory = "/data/Cityscapes"
    drn_name = 'drn_d_22' # 'drn_d_22' 'drn_d_38'
    weights = None
    if 'SL' in model_name:
        weights = {'offset_mean_weight' : 1e-5, #1e-3
                   'offset_variance_weight' : 1e-4, # 1e-3
                   'disparity_mean_weight' : 1e-7, #1e-3
                   'disparity_variance_weight' : 1e-4 }# 1e-3
    output_directory = "tmp/train/{}".format(model_name)
    #output_directory = "tmp/train/{}_{}"\
    #                    .format(model_name, time.strftime('%m%d-%H%M'))
    #output_directory = "tmp/train_test"
    #output_directory = "tmp/train_combined"
    #raise ValueError("Please set the input/output directories.")
    print("batch_size =",batch_size)
    print("train_split =",train_split)
    print("val_split =",val_split)
    print(locals())

    check_mkdir(output_directory)

    checkpoint = None

    check_mkdir(output_directory)

    checkpoint = None
    #checkpoint = (
    #        "/home/thomashehn/Code/box2pix/tmp/train/models/Net_epoch6.pth",
    #        "/home/thomashehn/Code/box2pix/tmp/train/models/Adam_epoch6.pth",
    #        6)

    n_classes = 19
    mdl = ModelWrapper(model_name, n_classes, weights, drn_name)

    #for param in parameters:
    #    param.require_grad = False
    #parameters = []
    # weight_decay=1e-6 seems to work so far, but would need more finetuning
    optimizer = optim.Adam(mdl.parameters, weight_decay=1e-6)
    start_epoch = 1

    if checkpoint is not None:
        print("Loading from checkpoint {}".format(checkpoint))
        mdl.NN.load_state_dict(torch.load(checkpoint[0]))
        optimizer.load_state_dict(torch.load(checkpoint[1]))
        start_epoch = checkpoint[2]+1

    mdl.NN, optimizer = amp.initialize(mdl.NN, optimizer, opt_level="O1")
    # O0, DRNDSDoubleSegSL, bs 6, cropped, 2 epochs -> 11949MB memory, time real 19m34.788s
    # O1, DRNDSDoubleSegSL, bs 6, cropped, 2 epochs -> 7339MB memory, time real 10m32.431s

    # O0, DRNDSOffsetDisparity, bs 6, cropped, 2 epochs -> 11875MB memory, time real 18m13.491s
    # O1, DRNDSOffsetDisparity, bs 6, cropped, 2 epochs -> 7259MB memory, time real 8m51.849s
    # O0, DRNDSOffsetDisparity, bs 7, cropped, 2 epochs -> memory error
    # O1, DRNDSOffsetDisparity, bs 7, cropped, 2 epochs -> 8701MB memory, time real 9m13.947s
    # O2, DRNDSOffsetDisparity, bs 7, cropped, 2 epochs -> 8721MB memory, time real 9m8.563s
    # O3, DRNDSOffsetDisparity, bs 7, cropped, 2 epochs -> 8693MB memory, time real 9m7.476s

    print("--- Setup dataset and dataloaders.")
    mdl.train_set =\
            Cityscapes(mdl.types,
                       data_split=train_split,
                       length=train_set_length,
                       cityscapes_directory=cityscapes_directory,
                       **mdl.dataset_kwargs)
    element = mdl.train_set[0]
    mdl.train_loader = data.DataLoader(mdl.train_set, batch_size=batch_size,
                                       pin_memory=True, num_workers=n_workers,
                                       shuffle=True)

    if not validate_on_train:
        mdl.val_set = Cityscapes(mdl.types,
                                 data_split=val_split,
                                 cityscapes_directory=cityscapes_directory,
                                 **mdl.val_dataset_kwargs)
    else:
        mdl.val_set =\
                Cityscapes(mdl.types,
                           data_split=train_split,
                           length=train_set_length,
                           cityscapes_directory=cityscapes_directory,
                           **mdl.val_dataset_kwargs)
    mdl.val_loader = data.DataLoader(mdl.val_set, batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=n_workers)

    # Sample 10 validation indices for visualization.
    #validation_idxs = np.random.choice(np.arange(len(val_set)),
    #                                   size=min(9, len(val_set)),
    #                                   replace=False)
    # Nah, let's pick them ourselves for now.
    #validation_idxs = [ 17, 241, 287, 304, 123,
    #                   458,   1,  14, 139, 388]
    validation_idxs = [ 17,  1,  14]
    #validation_idxs = [ 53,  11,  77]

    metrics = {'train' : {'classification' : [],
                          'regression' : [],
                          'epochs' : []},
               'validation' : {'classification' : [],
                               'regression' : [],
                               'semantic' : [],
                               'epochs' : []},
               'memory' : {'max_cached' : [torch.cuda.max_memory_cached()],
                           'max_alloc' : [torch.cuda.max_memory_allocated()]} }

    if initial_validation:
        print("--- Setup visual validation.")
        model_file = mdl.save_model(output_directory, suffix="e0000")
        mdl.validation_visual(validation_idxs, output_directory, epoch=0)
        semantic_score =\
                mdl.validation_snapshot(model_file,
                                        path.join(output_directory,
                                                  'last_prediction'),
                                        cityscapes_directory,
                                        batch_size, val_split)
        train_losses = mdl.compute_loss(mdl.train_loader)
        val_losses = mdl.compute_loss(mdl.val_loader, separate=True)
        print('Training loss: {:5} (c) + {:5} (r) = {:5}'
              .format(train_losses[0], train_losses[1], sum(train_losses)))
        if len(val_losses) > 5:
            val_dict = { 'offset_mean_loss' : [val_losses[2]],
                         'offset_variance_loss' : [val_losses[3]],
                         'disparity_mean_loss' : [val_losses[4]],
                         'disparity_variance_loss' : [val_losses[5]] }
        print('Validation loss: {:5} (c) + {:5} (r) = {:5}'
              .format(val_losses[0], val_losses[1], sum(val_losses[:2])))
        metrics = {'train' :
                    {'classification' : [train_losses[0]],
                     'regression' : [train_losses[1]],
                     'epochs' : [start_epoch-1]},
                  'validation' :
                    {'classification' : [val_losses[0]],
                     'regression' : [val_losses[1]],
                     'semantic' : [semantic_score],
                     'epochs' : [start_epoch-1],
                     **val_dict },
                  'memory' : {'max_cached' : [torch.cuda.max_memory_cached()],
                              'max_alloc' : [torch.cuda.max_memory_allocated()]
                             } }

    print("--- Training.")
    # First train semantic loss for a while.
    #~regression_loss_stash = None
    #~if n_semantic_pretrain > 0 and regression_loss is not None:
    #~    regression_loss_stash = regression_loss
    #~    regression_loss = None
    #upscale = lambda x: nn.functional.interpolate(x,
    #                        scale_factor=2,
    #                        mode='bilinear',
    #                        align_corners=True)
    for epoch in range(start_epoch, n_epochs+1):
        #~if epoch >= n_semantic_pretrain and regression_loss_stash is not None:
        #~    regression_loss = regression_loss_stash
        #~    regression_loss_stash = None

        #~if epoch == 10 and False:
        #~    parameters = model.parameters()
        #~    model.train_all = True
        #~    optimizer = optim.Adam(parameters)

        mdl.NN.train()
        total_rloss = 0
        total_closs = 0

        t_sum_batch = 0
        t_sum_opt = 0
        for batch_idx, batch_data in enumerate(mdl.train_loader):
            optimizer.zero_grad()

            batch_losses = mdl.batch_loss(batch_data)

            loss = sum(batch_losses)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            #loss.backward()
            optimizer.step()

            if batch_idx % 30 == 0 and batch_idx != 0:
                print('\t[batch {}/{}], [batch mean - closs {:5}, rloss {:5}]'
                      .format(batch_idx, len(mdl.train_loader),
                              float(batch_losses[0])/batch_data[0].size(0),
                              float(batch_losses[1])/batch_data[0].size(0)))

            total_closs += float(batch_losses[0])
            total_rloss += float(batch_losses[1])
        del loss, batch_data, batch_losses

        total_closs /= len(mdl.train_set)
        total_rloss /= len(mdl.train_set)

        print('[epoch {}], [mean train - closs {:5}, rloss {:5}]'
              .format(epoch, total_closs, total_rloss))
        metrics['train']['classification'].append(total_closs)
        metrics['train']['regression'].append(total_rloss)
        metrics['train']['epochs'].append(epoch)

        metrics['memory']['max_cached'].append(torch.cuda.max_memory_cached())
        metrics['memory']['max_alloc'].append(torch.cuda.max_memory_allocated())

        # --- Visual validation.
        if (epoch % validation_step) == 0:
            print("--- Validation.")
            mdl.validation_visual(validation_idxs, output_directory, epoch)

            model_file = mdl.save_model(output_directory,
                                        suffix="{:04}".format(epoch))
            metrics['validation']['semantic'].append(
                    mdl.validation_snapshot(
                            model_file,
                            path.join(output_directory, 'last_prediction'),
                            cityscapes_directory,
                            batch_size, val_split))

            val_losses = mdl.compute_loss(mdl.val_loader, separate=True)
            if len(val_losses) > 5:
                if 'offset_mean_loss' not in metrics['validation'].keys():
                    val_dict = {'offset_mean_loss' : [val_losses[2]],
                                'offset_variance_loss' : [val_losses[3]],
                                'disparity_mean_loss' : [val_losses[4]],
                                'disparity_variance_loss' : [val_losses[5]]}
                    metrics['validation'] = {**metrics['validation'],
                                             **val_dict}
                else:
                    metrics['validation']['offset_mean_loss']\
                            .append(val_losses[2])
                    metrics['validation']['offset_variance_loss']\
                            .append(val_losses[3])
                    metrics['validation']['disparity_mean_loss']\
                            .append(val_losses[4])
                    metrics['validation']['disparity_variance_loss']\
                            .append(val_losses[5])
                print('Separate validation losses: {:5}, {:5}, {:5}, {:5}'
                      .format(*val_losses[2:]))

            metrics['validation']['classification'].append(val_losses[0])
            metrics['validation']['regression'].append(val_losses[1])
            metrics['validation']['epochs'].append(epoch)
            print('Validation loss: {:5} (c) + {:5} (r) = {:5}'
                  .format(val_losses[0], val_losses[1], sum(val_losses[:2])))

        # --- Write losses to disk.
        with open(path.join(output_directory, "metrics.json"),'w') as outfile:
            json.dump(metrics, outfile)
        for key in metrics.keys():
            data_set = key
            set_metrics = metrics[data_set]
            plot_losses(set_metrics, "{}/{}".
                        format(output_directory, data_set))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description="Train a segmentation and offset model.")
    parser.add_argument("--model", type=str,
                        help="Choose a model.")
    parser.add_argument("--initial-validation", "-i", action="store_true",
                        help="Run validation on initial model.")
    args = parser.parse_args()

    main(args.model, args.initial_validation)
