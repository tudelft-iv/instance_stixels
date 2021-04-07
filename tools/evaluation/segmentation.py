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

# DO NOT USE! This is buggy and gives much lower results. For now, use instead:
# python -m cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling\
# data/cityscapes_val/gtFine/val/dummy/*labelIds.png\
# data/cityscapes_val/pred/../results_w0.0/segmentation_results/

import os
import argparse

from cityscapesscripts.evaluation import evalPixelLevelSemanticLabeling

def evaluate_semanticsegmentation(prediction_directory, groundtruth_directory):
    prediction_files = [os.path.join(prediction_directory, f)\
                        for f in os.listdir(prediction_directory)]
    groundtruth_files = [os.path.join(groundtruth_directory, f)\
                         for f in os.listdir(groundtruth_directory)\
                         if "_labelIds.png" in f]
    
    print(prediction_files)
    print(groundtruth_files)
    evalPixelLevelSemanticLabeling.evaluateImgLists(prediction_files,
            groundtruth_files, evalPixelLevelSemanticLabeling.args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Script to evaluate the semantic segmentation on "\
            "Cityscapes.")
    parser.add_argument(
            "PREDICTIONS", type=str, 
            help="Directory containing the predicted semantic segmentation.")
    parser.add_argument(
            "GROUNDTRUTH", type=str, 
            help="Directory containing the groundtruth semantic segmentation.")

    args = parser.parse_args()

    evaluate_semanticsegmentation(args.PREDICTIONS, args.GROUNDTRUTH)
