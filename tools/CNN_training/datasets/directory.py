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

import glob

import PIL

from torch.utils import data
from torchvision import transforms

class Directory(data.Dataset):
    """
    A simple dataset class that allows to read all *.png files from a
    directory.
    """
    def __init__(self, directory,
                 pil_transforms=None,
                 tensor_transforms=None,
                 input_transforms=None, # same as tensor_trfs for this class
                 suffix=".png", mode='RGB', use_ENet=False):
        self.use_ENet = use_ENet
        self.mode = mode
        self.directory = directory

        self.filenames = []
        self.filenames += glob.glob('{}/*{}'.format(self.directory, suffix))
        self.filenames += glob.glob('{}/*/*{}'.format(self.directory, suffix))
        if len(self.filenames) == 0:
            raise IOError("No matching files found in {}."
                          .format(self.directory))
        self.filenames.sort()

        # Setup input transformations. Input and tensor transforms are handled
        # the same way as this loader does not support groundtruth.
        self.pil_transforms = pil_transforms or []
        self.tensor_transforms = tensor_transforms or input_transforms or []
        print("Tensor transfroms {}".format(self.tensor_transforms))
        if transforms.ToTensor() not in self.tensor_transforms:
            self.tensor_transforms =\
                    [transforms.ToTensor()] + self.tensor_transforms
        self.tensor_transforms = transforms.Compose(self.tensor_transforms)

        self.padding_size = None
        self.check_input()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img = self._load_PIL_images(index)
        return self.filenames[index], self.tensor_transforms(img)

    def _load_PIL_images(self, index):
        img = PIL.Image.open(self.filenames[index])
        if self.mode == 'RGB':
            img = img.convert('RGB')
        return self.pil_transforms(img)

    def check_input(self, index=0):
        # Separate these to add stuff if necessary.
        pil_transforms = self.pil_transforms
        self.pil_transforms = transforms.Compose(self.pil_transforms)

        img = self._load_PIL_images(index)
        if self.use_ENet and (img.size[0] % 8 != 0 or img.size[1] % 8 != 0):
            padding_size = (0,0, # left, top
                            (8 - img.size[0] % 8) % 8, # right
                            (8 - img.size[1] % 8) % 8) # bottom
            pil_transforms.append(transforms.Pad(padding_size, fill=0))
            self.pil_transforms = transforms.Compose(pil_transforms)
            self.padding_size = padding_size
            warnings.warn("Padding input images with {} x {} x {} x {} zeros "
                          "(left x top x right x bottom) since this "
                          "ENet implementation requires dimension to be "
                          "multiples of 8."
                          .format(*padding_size))

