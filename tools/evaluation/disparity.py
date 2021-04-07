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
import argparse

import numpy as np
import PIL.Image

#import ipdb

def print_disparity_evaluation(disparity_folder1, disparity_folder2):
    means = evaluate_disparity(disparity_folder1, disparity_folder2)
    print("Mean deviation: {}".format(means.mean()))
    print("Mean deviation / 256: {}".format(means.mean() / 256))
    return means

def evaluate_disparity(disparity_folder1, disparity_folder2):
    """
    This function reads all files in the given folders. Compares the length,
    sorts them and opens them as images. The root mean squared deviation is
    computed for each image pair. This servers as metric for disparity
    estimations.
    """
    disparity_files1 = [os.path.join(disparity_folder1,f)\
                        for f in sorted(os.listdir(disparity_folder1))]
    disparity_files2 = [os.path.join(disparity_folder2,f)\
                        for f in sorted(os.listdir(disparity_folder2))]

    if len(disparity_files1) != len(disparity_files2):
        raise IOError("File lists are not of the same length: {} != {}".\
                format(disparity_files1, disparity_files2))

    means = []
    for file1, file2 in zip(disparity_files1, disparity_files2):
        #stem_name = re.sub('_[a-z].*','',os.path.basename(f))
        print("Comparing {} and {}".format(file1,file2))
        disparity_img1 = np.array(PIL.Image.open(file1)).astype(np.float)
        disparity_img2 = np.array(PIL.Image.open(file2)).astype(np.float)

        diff_img = np.sqrt((disparity_img1 - disparity_img2)**2)
        # TODO: Use only original disparity for mask?
        # Actually, I think this is not important. If stixel disparity is 0
        # then also original disparity was 0 in the entire stixel.
        non_zero_mask =  (disparity_img1 != 0).astype(np.bool)\
                       & (disparity_img2 != 0).astype(np.bool)
        mean = np.mean(diff_img[non_zero_mask])
        means.append(mean)

    return np.array(means)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Script to compare two disparity images.")
    parser.add_argument(
            "FOLDER1", type=str,
            help="First folder with disparity files.")
    parser.add_argument(
            "FOLDER2", type=str,
            help="Second folder with disparity files.")
    args = parser.parse_args()
    means = print_disparity_evaluation(args.FOLDER1, args.FOLDER2)

