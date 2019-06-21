#!/usr/bin/env python

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

import numpy as np
import torch
from torchvision import transforms
import PIL.Image
from matplotlib import pyplot as plt

import cityscapesscripts.helpers.labels as cityscapes_labels
#import ipdb

def save_cityscapesprediction(
        prediction, filename, colored=False, resize=None):
    """
    Takes `predictions` (i.e. argmax of probabilites), translates trainIds to
    ids and saves them in file `filename`.
    """
    prediction = prediction.cpu().byte()

    if not colored: 
        # --- Save as labelImg with labelIds.
        if "_leftImg8bit.png" in filename:
            filename = filename.replace("_leftImg8bit.png", "_predlabelImg.png")

        trainIds = range(19)
        # Reversed to avoid overwriting of ids. trainId <= id.
        for trainId in reversed(trainIds):
            id_ = cityscapes_labels.trainId2label[trainId].id
            prediction[prediction == trainId] = id_

        if prediction.dim() < 3:
            prediction = prediction.unsqueeze(dim=0)
        prediction_pil = transforms.ToPILImage()(prediction)
    else:
        # --- Create a fancy colorful prediction image.
        prediction_pil = PIL.Image.fromarray(prediction.numpy(),
                                             mode='P')
        palette = np.array(cityscapes_labels.trainId2color).astype(np.uint8)
        palette.resize((256,3))
        prediction_pil.putpalette(palette)

    if resize is not None:
        prediction_pil = prediction_pil.resize(resize, PIL.Image.NEAREST)
    prediction_pil.save(filename)

def visualize_positionplusoffset(mapping, filename, groundtruth=None):
    """
    Visualize the predicted offset plus the pixel position in a 2D scatter 
    plot. 
    mapping : Mapping obtained for a single image. 
        Size: (batchsize=1 x) mapping dimension x height x width
    """
    # Get rid of auto diff and transfer to CPU/RAM.
    mapping = mapping.detach().cpu()
    # Remove single sample batch dimension, if necessary.
    if mapping.dim() == 4:
        mapping = mapping[0]

    positions = torch.meshgrid([torch.arange(mapping.size(1)),
                                torch.arange(mapping.size(2))])
    positions = torch.stack(positions).float()

    sum_th = positions + mapping[-2:]
    # If you have mean positions in the mapping then add this.
    #sum_th[mapping != 0] = mapping[mapping != 0]

    plt.figure(figsize=(20,12))
    # s <= 0.5 does not work for what ever reason...
    if groundtruth is None:
        sum_np = sum_th.numpy().reshape(2,-1)
        plt.scatter(                  sum_np[1,:], 
                    mapping.size(1) - sum_np[0,:], 
                    s=0.55, marker="s")
    else:
        # 0 if no offset in any dimension
        nonzero = (groundtruth[0] == 0) * (groundtruth[1] == 0)
        zerosum_np = sum_th[:, ~nonzero].numpy().reshape(2,-1)
        nonzerosum_np = sum_th[:, nonzero].numpy().reshape(2,-1)
        plt.scatter(                  nonzerosum_np[1,:], 
                    mapping.size(1) - nonzerosum_np[0,:], 
                    s=0.55, marker="s", c='blue')
        plt.scatter(                  zerosum_np[1,:], 
                    mapping.size(1) - zerosum_np[0,:], 
                    s=0.55, marker="s", c='red')
    plt.ylim([0, mapping.size(1)])
    plt.xlim([0, mapping.size(2)])
    plt.axis('equal')
    plt.savefig('{}.png'.format(filename))
    plt.close('all')

def visualize_offsethsv(mapping, filename):
    """
    Visualize the predicted offset plus the pixel position in a 2D scatter 
    plot. 
    mapping : Mapping obtained for a single image. 
        Size: (batchsize=1 x) mapping dimension x height x width
    """
    # Get rid of auto diff.
    # Remove single sample batch dimension, if necessary.
    if mapping.dim() == 4:
        mapping = mapping[0]
    if mapping.size()[0] > 2:
        mapping = mapping[-2:].detach()

    magnitude = torch.sqrt(torch.sum(mapping**2, dim=0))
    orientation = torch.acos(mapping[1] / magnitude)
    orientation[mapping[0] < 0] *= -1

    # Scale and clip it.
    magnitude = torch.clamp(magnitude*8 / mapping.size()[2], 0, 1) * 255
    magnitude = np.uint8(magnitude.cpu().numpy())

    orientation = (orientation + np.pi) / (2*np.pi) * 255
    orientation = np.uint8(orientation.cpu().numpy())

    hsv_np = np.stack((orientation, magnitude, np.ones_like(magnitude)*255),
                      axis=2)
    hsv_pil = PIL.Image.fromarray(hsv_np, mode='HSV')
    hsv_pil.convert('RGB').save('{}.png'.format(filename))

def visualize_semantics(img, mapping, filename, overlay_filename=None):
    """
    TODO
    """
    mapping = mapping.detach().cpu()
    # Remove single sample batch dimension, if necessary.
    if mapping.dim() == 4:
        mapping = mapping[0]

    if type(img) == torch.Tensor:
        img = img.detach().cpu()
        if img.dim() == 4:
            img = img[0]

        img_pil = transforms.ToPILImage()(img)
        #img_pil = PIL.Image.fromarray(img.numpy(), mode='RGB')
    elif type(img) == PIL.Image.Image:
        img_pil = img
    else:
        raise TypeError("Unkown data type {} for image."
                        .format(type(img)))

    # Handle groundtruth as well as network output.
    if mapping.dim() == 3:
        mapping = mapping[:-2].argmax(dim=0).long()

    prediction_pil = PIL.Image.fromarray(mapping.numpy().astype(np.uint8), 
                                         mode='P')
    palette = np.array(cityscapes_labels.trainId2color).astype(np.uint8)
    palette.resize((256,3))
    prediction_pil.putpalette(palette)

    prediction_pil = prediction_pil.convert(mode="RGB")
    prediction_pil.save('{}_pred.png'.format(filename))

    if overlay_filename is None:
        overlay_filename = filename
    img_pil = PIL.Image.blend(img_pil, prediction_pil, 0.6)
    img_pil.save('{}_overlay.png'.format(overlay_filename))

