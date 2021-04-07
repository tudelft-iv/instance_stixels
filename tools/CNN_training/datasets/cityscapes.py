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

import os
import glob
import re
import json
import PIL.Image
import torch
from torch.utils import data
import torchvision.transforms

import cityscapesscripts.helpers.labels as cityscapes_labels

class Cityscapes(data.Dataset):
    """
    Dataset specialized to collect files from cityscape-like directory/
    file structure.
    """
    # TODO: Transforms are not synchornized. Transforms with random
    # elements may be different per image.
    def __init__(self, types, data_split="subtrain", cityscapes_directory=None,
                 transforms=None, length=None):
        self.data_split = data_split
        assert self.data_split in ['subtrain', 'subtrainval', 'train', 'val']
        self.length = length
        self.image_mesh = None

        self.transforms = transforms
        if transforms is None:
            self.transforms = {
                    'tensor' : {
                        'pre' : torchvision.transforms.ToTensor()}}
        elif isinstance(transforms, torchvision.transforms.Compose):
            self.transforms = {
                    'tensor' : {
                        'pre' : torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    transforms])} }
        elif ('tensor' not in transforms.keys()
              or 'pre' not in transforms['tensor'].keys()):
            raise ValueError("Incorrect use of transforms.")

        # Set cityscapes base directory.
        self.cityscapes_directory = cityscapes_directory
        if self.cityscapes_directory is None:
            if os.env['CITYSCAPES_PATH']:
                self.cityscapes_directory = os.env['CITYSCAPES_PATH']
            else:
                raise ValueError("Cityscapes directory is not set.")

        self.types = tuple(types)
        # Collect paths to required png files.
        self._collect_cityscapes_files()

    def _collect_cityscapes_files(self):
        """
        Find left images, semantic groundtruth and instance groundtruth
        in the Cityscapes directory and store them as members.
        """
        type_dict = {
            'left'            : '{}/leftImg8bit/{}/*/*_leftImg8bit.png',
            'semantic_gt'     : '{}/gtFine/{}/*/*_gtFine_labelTrainIds.png',
            'instance_gt'     : '{}/gtFine/{}/*/*_gtFine_instanceTrainIds.png',
            'instance_gt_raw' : '{}/gtFine/{}/*/*_gtFine_instanceTrainIds.png',
            'camera'          : '{}/camera/{}/*/*_camera.json',
            'disparity'       : '{}/disparity/{}/*/*_disparity.png',
            'disparity_raw'   : '{}/disparity/{}/*/*_disparity.png' }
        self.paths = {}
        for type_ in self.types:
            self.paths[type_] =\
                glob.glob(type_dict[type_]
                          .format(self.cityscapes_directory, self.data_split))
            self.paths[type_].sort()

        # Check for consistent number of collected paths.
        lengths = {type_ : len(paths) for type_, paths in self.paths.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise IOError("Number of input files in cityscapes directory does "
                          "not match.\n"
                          "Cityscapes directory: {}"
                          "Number of files = {}, "
                          "Glob strings = {}, "
                          .format(self.cityscapes_directory,
                                  lengths,
                                  type_dict))
        if 0 in unique_lengths:
            raise IOError("No matching files found in {}."
                          .format(self.cityscapes_directory))

        # In case we want to use a smaller dataset.
        if self.length is not None:
            self.paths = {type_ : paths[:self.length]\
                          for type_, paths in self.paths.items()}
        else:
            self.length = unique_lengths.pop()

    def _instance_offsets_disparity(self, instance_gt, disparity_img):
        """
        Compute offset per pixel to center of mass of corresponding instance
        mask as well as the median disparity of this instance mask.
        """
        offsets = torch.zeros((3,*instance_gt.size())) # 3 x h x w
        disparity_img = disparity_img // 256

        # Get all relevant instance labels (id > 1000).
        instance_ids = torch.unique(instance_gt)
        instance_ids = instance_ids[instance_ids > 1000]
        # Compute center of mass of each instance mask and per pixel offsets as
        # well as median disparity.
        for instance_id in instance_ids:
            binary_mask = (instance_gt == instance_id)
            positions_ind = torch.nonzero(binary_mask).t()
            positions = positions_ind.float()
            disparities = disparity_img[positions_ind[0],positions_ind[1]]
            disparities = disparities[disparities != 0]

            center_of_mass = positions.mean(dim=1).reshape(2,1)
            median_disparity = 0
            if len(disparities) > 0:
                median_disparity = disparities.median()
            # Using the positions is faster than binary_mask.
            offsets[0,positions_ind[0],positions_ind[1]] =\
                    median_disparity
            offsets[1:,positions_ind[0],positions_ind[1]] =\
                    center_of_mass - positions

        return offsets

    def _instance_offsets(self, instance_gt):
        """
        Compute offset per pixel to center of mass of corresponding instance
        mask.
        """
        offsets = torch.zeros((2,*instance_gt.size())) # 2 x h x w

        # Get all relevant instance labels (id > 1000).
        instance_ids = torch.unique(instance_gt)
        instance_ids = instance_ids[instance_ids > 1000]
        # Compute center of mass of each instance mask and per pixel offsets.
        for instance_id in instance_ids:
            binary_mask = (instance_gt == instance_id)
            positions_ind = torch.nonzero(binary_mask).t()
            positions = positions_ind.float()

            center_of_mass = positions.mean(dim=1).reshape(2,1)
            # Using the positions is faster than binary_mask.
            offsets[:,positions_ind[0],positions_ind[1]] =\
                    center_of_mass - positions

        return offsets

    def _load_paths(self, index):
        # Check if image names match.
        re_pattern = re.compile("([a-z]*)_([0-9]*)_([0-9]*)_")
        paths = { type_ : paths[index]\
                  for type_, paths in self.paths.items() }
        if len(set([re_pattern.search(path).groups()\
                    for path in paths.values()])) > 1:
            raise ValueError("Paths to do not match: {}"
                             .format(paths))

        return paths

    def _load_PIL_images(self, paths):
        # TODO: Think about using accimage from pytorch for more efficent image
        # loading.
        images = { type_ : PIL.Image.open(path)\
                   for type_, path in paths.items()\
                   if path.endswith(".png") }

        default_preprocessing = {'left' : lambda img: img.convert()}

        for type_, action in default_preprocessing.items():
            if type_ in images.keys():
                images[type_] = action(images[type_])

        # Apply transformations.
        if 'pil' in self.transforms.keys():
            if 'pre' in self.transforms['pil'].keys():
                images = {type_ : self.transforms['pil']['pre'](image)\
                          for type_, image in images.items()}
            if 'combined' in self.transforms['pil'].keys():
                # Note: for the combined transform we need to pass a list and
                # make sure returned images are associated to the right key.
                types = list(images.keys())
                images_list = [images[type_] for type_ in types]
                images_list = self.transforms['pil']['combined'](images_list)
                images = {type_ : image\
                          for type_, image in zip(types, images_list)}

            for type_, trf in self.transforms['pil'].items():
                if type_ in images.keys():
                    images[type_] = trf(images[type_])

        return images # img, semantic_gt, instance_gt

    def _image_preprocessing(self, images):
        images = {type_ : self.transforms['tensor']['pre'](image)\
                  for type_, image in images.items()}

        # --- Default preprocessing, such as offset computation.
        # TODO: cache if possible.
        instance_preprocessing =\
                lambda img: self._instance_offsets(img.squeeze())
        if 'disparity' in images.keys():
            instance_preprocessing = lambda img:\
                        self._instance_offsets_disparity(
                                img.squeeze(), images['disparity'].squeeze())
        default_preprocessing = {
                'semantic_gt' : lambda img: (img * 255).long().squeeze(),
                'instance_gt' : instance_preprocessing,
                'disparity_raw' : lambda img: img // 256}
        for type_, action in default_preprocessing.items():
            if type_ in images.keys():
                images[type_] = action(images[type_])

        # --- Apply user tensor transformations.
        for type_, trf in self.transforms['tensor'].items():
            if type_ in images.keys():
                images[type_] = trf(images[type_])

        return images

    def get_images(self, index):
        paths = self._load_paths(index)
        images = self._load_PIL_images(paths)
        return images

    def _load_json(self, paths):
        json_data = {}
        for type_, path in paths.items():
            if path.endswith(".json"):
                with open(path, 'r') as json_file:
                    json_data[type_] = json.load(json_file)

        return json_data

    def __getitem__(self, index):
        paths = self._load_paths(index)
        # - Process images.
        images = self._load_PIL_images(paths)
        images = self._image_preprocessing(images)

        # - Process json.
        json_data = self._load_json(paths)

        # Produce a ordered tuple.
        unordered_data = {**images, **json_data}
        return tuple(unordered_data[type_] for type_ in self.types)

    def __len__(self):
        return self.length
