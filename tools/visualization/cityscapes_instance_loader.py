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

import os
import numpy as np
from PIL import Image

num_classes = 19
ignore_label = 255

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def load_instance_mask(mask_path, return_trainIds=True):
    # generate c*w*h instance masks
    mask = Image.open(mask_path)
    mask = np.array(mask)

    # Remove non-instance classes (IDs < 1000).
    mask_both = mask * (mask > 1000)
    r, c = mask_both.shape

    # Train IDs (person, rider, car, truck, bus, train, motorcycle, bicycle)
    train_ids = np.array([11,12,13,14,15,16,17,18])
    if mask_path.endswith("_instanceIds.png"):
        # IDs (see cityscapes labels.py for difference) - same classes
        classes = np.array([24,25,26,27,28,31,32,33])
    elif mask_path.endswith("_instanceTrainIds.png"):
        classes = train_ids
    else:
        raise IOError("No matching filename ending found. Expected "\
                "\"_instanceIds.png\" or \"_instanceTrainIds.png\".")
    n = len(classes)
    # Instance mask per class.
    masks_per_class = np.empty((n,r,c)) # classes x rows x cols
    idx = 0

    # --- Create masks per class.
    for idx, (id_, train_id) in enumerate(zip(classes, train_ids)):
        mask_class = mask_both\
                     * (mask_both >= id_ * 1000)\
                     * (mask_both < (id_+1) * 1000)
        # Indistinguishable instances however get id = i*1000, which means
        # a simple modulo is not enough therefore we keep the ids as they
        # are or convert them to train_id.
        if return_trainIds and id_ != train_id:
            non_zero = mask_class != 0
            # e.g. car (id=26, train_id=13) instance 1: 26001 -> 13001
            mask_class[non_zero] = mask_class[non_zero] % (id_ * 1000)\
                                   + train_id * 1000
        masks_per_class[idx] = mask_class

    return masks_per_class

