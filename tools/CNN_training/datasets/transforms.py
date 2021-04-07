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

import random
import torch
from torchvision.transforms import functional as F
from PIL import Image
#from PIL import ImageFilter, ImageChops

#import ipdb

# From: https://github.com/pytorch/vision/blob/50d54a8/references/segmentation/transforms.py
class MultiImgRandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, images):
        if random.random() < self.flip_prob:
            images = [F.hflip(image) for image in images]
        return images

# From: https://github.com/pytorch/vision/blob/50d54a8/references/segmentation/transforms.py
class MultiImageCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        for t in self.transforms:
            images = t(images)
        return images

class ModeDownsample(object):
    def __init__(self, factor):
        self.factor = factor
        if self.factor % 2 != 0:
            raise ValueError("Only tested for even factors.")

    def __call__(self, img):
        result = self.modefilter_np(np.array(img), self.factor)
        return Image.fromarray(result)

    def modefilter_np(self, img_np, factor):
        """
        Only use with even factors. This could also be jitted with numba I
        think.
        """
        result = np.empty(np.array(img_np.shape) // factor, dtype=img_np.dtype)
        for i in range(0, img_np.shape[0], factor):
            for j in range(0, img_np.shape[1], factor):
                slice_ = img_np[i:i+factor, j:j+factor]
                result[i//factor, j//factor] =\
                        np.bincount(np.ravel(slice_)).argmax()
        return result

    #def __call__(self, img):
    #    # This does only work for img.mode == 'L' (i.e. uint8, grayscale).
    #    if img.size[0] % self.factor != 0 or img.size[1] % self.factor != 0:
    #        raise ValueError("Only tested for images divisible by factor.")

    #    #ipdb.set_trace()
    #    new_size = (img.size[0] // self.factor,
    #                img.size[1] // self.factor)

    #    img = img.filter(ImageFilter.ModeFilter(self.factor))
    #    img = ImageChops.offset(img, 1, 1)
    #    img = img.resize(new_size, Image.NEAREST)
    #    return img

    def __repr__(self):
        return self.__class_.__name__ + "(factor={})".format(self.factor)
