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
import PIL.Image
import torch
from torch.utils import data
from torchvision import transforms

import cityscapesscripts.helpers.labels as cityscapes_labels

class Cityscapes(data.Dataset):
    """
    Dataset specialized to collect files from cityscape-like directory/
    file structure.
    """
    # TODO: Add transformation option
    def __init__(self, data_split="subtrain", cityscapes_directory=None,
                 pil_transforms=None, 
                 gt_pil_transforms=None,
                 tensor_transforms=None,
                 fit_gt_pil_transforms=None,
                 input_transforms=None, # transforms only on input, not target
                 length=None):
        self.data_split = data_split
        assert self.data_split in ['subtrain', 'subtrainval']
        self.length = length

        # --- Setup input transformations.
        self.pil_transforms = lambda x: x
        if pil_transforms is not None:
            self.pil_transforms = transforms.Compose(pil_transforms)

        self.gt_pil_transforms = lambda x: x
        if gt_pil_transforms is not None:
            self.gt_pil_transforms = transforms.Compose(gt_pil_transforms)

        self.fit_gt_pil_transforms = lambda x: x
        if fit_gt_pil_transforms is not None:
            self.fit_gt_pil_transforms =\
                    transforms.Compose(fit_gt_pil_transforms)

        self.tensor_transforms = [transforms.ToTensor()]
        if tensor_transforms is not None: 
            self.tensor_transforms += tensor_transforms
        self.tensor_transforms = transforms.Compose(self.tensor_transforms)

        self.input_transforms = lambda x: x
        if input_transforms is not None:
            self.input_transforms = transforms.Compose(input_transforms)

        # Set cityscapes base directory.
        self.cityscapes_directory = cityscapes_directory
        if self.cityscapes_directory is None:
            if os.env['CITYSCAPES_PATH']:
                self.cityscapes_directory = os.env['CITYSCAPES_PATH']
            else:
                raise ValueError("Cityscapes directory is not set.")

        # Collect paths to required png files.
        self._collect_cityscapes_files()

    def _collect_cityscapes_files(self):
        """
        Find left images, semantic groundtruth and instance groundtruth 
        in the Cityscapes directory and store them as members.
        """
        self.left_img_paths =\
                glob.glob('{}/leftImg8bit/{}/*/*_leftImg8bit.png'
                          .format(self.cityscapes_directory, self.data_split))
        self.left_img_paths.sort()

        self.semantic_gt_paths =\
                glob.glob('{}/gtFine/{}/*/*_gtFine_labelTrainIds.png'
                          .format(self.cityscapes_directory, self.data_split))
        self.semantic_gt_paths.sort()

        self.instance_gt_paths =\
                glob.glob('{}/gtFine/{}/*/*_gtFine_instanceTrainIds.png'
                          .format(self.cityscapes_directory, self.data_split))
        self.instance_gt_paths.sort()

        if not (len(self.left_img_paths)
                == len(self.semantic_gt_paths) 
                == len(self.instance_gt_paths)):
            raise IOError("Number of input files in cityscapes directory does "
                          "not match.\n"
                          "Cityscapes directory: {}"
                          "len(self.left_img_paths) = {}, "
                          "len(self.semantic_gt_paths) = {} and "
                          "len(self.instance_gt_paths) = {}"
                          .format(self.cityscapes_directory, 
                                  len(self.left_img_paths),
                                  len(self.semantic_gt_paths),
                                  len(self.instance_gt_paths)))
        if len(self.left_img_paths) == 0:
            raise IOError("No matching files found in {}."
                          .format(self.cityscapes_directory))

        if self.length is not None:
            self.left_img_paths = self.left_img_paths[:self.length]
            self.semantic_gt_paths = self.semantic_gt_paths[:self.length]
            self.instance_gt_paths = self.instance_gt_paths[:self.length]

    def compute_instance_offsets(self, instance_gt):
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
            instance_positions = torch.nonzero(binary_mask).float().t()

            center_of_mass = instance_positions.mean(dim=1).reshape(2,1)
            offsets[:,binary_mask] = center_of_mass - instance_positions

        return offsets

    def load_PIL_images(self, index):
        # TODO: Think about using accimage from pytorch for more efficent image
        # loading.
        re_pattern = re.compile("([a-z]*)_([0-9]*)_([0-9]*)_")
        left_img_path = self.left_img_paths[index]
        semantic_gt_path = self.semantic_gt_paths[index]
        instance_gt_path = self.instance_gt_paths[index]
        if not (re_pattern.search(left_img_path).groups()
                == re_pattern.search(semantic_gt_path).groups()
                == re_pattern.search(instance_gt_path).groups()):
            raise ValueError("Image paths to do not match: {}, {}, {}"
                             .format(left_img_path, 
                                     semantic_gt_path,
                                     instance_gt_path))

        img = PIL.Image.open(left_img_path).convert('RGB')
        semantic_gt = PIL.Image.open(semantic_gt_path)
        instance_gt = PIL.Image.open(instance_gt_path)

        img = self.pil_transforms(img)
        semantic_gt = self.pil_transforms(semantic_gt)
        instance_gt = self.pil_transforms(instance_gt)
        semantic_gt = self.gt_pil_transforms(semantic_gt)
        instance_gt = self.gt_pil_transforms(instance_gt)
        return img, semantic_gt, instance_gt

    def load_fit_gt_PIL_images(self, index):
        img, semantic_gt, instance_gt = self.load_PIL_images(index)
        img = self.fit_gt_pil_transforms(img)
        return img, semantic_gt, instance_gt

    def __getitem__(self, index):
        img, semantic_gt, instance_gt = self.load_PIL_images(index)

        img = self.input_transforms(self.tensor_transforms(img))
        # TODO: This seems a bit complicated...
        semantic_gt = self.tensor_transforms(semantic_gt).squeeze()
        semantic_gt = (semantic_gt * 255).long()
        instance_gt = self.tensor_transforms(instance_gt).squeeze()

        # TrainIds of instance classes.
        #classes = np.array([11,12,13,14,15,16,17,18])
        # Ids of instance classes.
        #classes = np.array([24,25,26,27,28,31,32,33])
        instance_offsets = self.compute_instance_offsets(instance_gt)

        return img, semantic_gt, instance_offsets
        
    def __len__(self):
        return len(self.left_img_paths)
