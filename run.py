#!/usr/bin/env python

# This file is part of Instance Stixels:
# https://github.com/tudelft-iv/instance-stixels
# This python script will automatically collect the required files, run the
# single components of the instance stixel pipeline. It does require a lot of
# subprocess calls. This may actually sound like it is a job for a bash script.
# Fair point. But then again: bash. Anyway, the bash script became too convoluted
# to maintain and bug prone.
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
import sys
import shutil
import glob
import warnings

import subprocess
import multiprocessing
import copy

import re
import time
import json
import argparse

import numpy as np
import PIL.Image

START_TIME = time.strftime('%m%d-%H%M')
CITYSCAPES_PATH = "/data/Cityscapes"

def parse_input_images(filedir):
    return [re.sub('_[a-z].*','',os.path.basename(f))\
            for f in os.listdir(filedir)]

def find_cityscapes_files(cityscapes_directory, stem_names, no_gt=False):
    """
    Find left images, disparity images, semantic groundtruth,
    instance groundtruth and camera files in the Cityscapes directory 
    corresponding to the stem_names provided.
    stem_names : List of Cityscapes stem names in format 
        "{city}_{seq:0>6}_{frame:0>6}".
    """
    cityscapes_paths = {
            'left' : [],
            'disparity' : [],
            'semanticgt' : [],
            'instancegt' : [],
            'camera' : []}
    CITYSCAPES_LOOKUP = {
            # type : [directory, fileending]
            'left' : ['leftImg8bit', 'leftImg8bit.png'],
            'disparity' : ['disparity', 'disparity.png'],
            #'disparity' : ['disparity_rossgm', 'disparity.png'],
            'semanticgt' : ['gtFine', 'gtFine_labelIds.png'],
            'instancegt' : ['gtFine', 'gtFine_instanceIds.png'],
            'camera' : ['camera', 'camera.json']}
    if no_gt:
        for key in list(cityscapes_paths.keys()):
            if key.endswith("gt"):
                print("Removing ground truth key {}.".format(key))
                del cityscapes_paths[key]

    for stem_name in stem_names:
        # glob for file of specific type. Error if none or multiple found.
        # glob according to {cityscapes_dir}/{type}/{split}/{city}/...
        def glob_one(type_):
            paths = glob.glob('{}/{}/*/*/{}_{}'.\
                              format(cityscapes_directory, 
                                     CITYSCAPES_LOOKUP[type_][0],
                                     stem_name,
                                     CITYSCAPES_LOOKUP[type_][1]))
            if len(paths) == 0:
                raise IOError("No matching {} file found for {}.".\
                              format(type_, stem_name))
            elif len(paths) > 1:
                warnings.warn("Warning: Multiple files found for {}:\n{}"\
                              "Returning only first one.".\
                              format(stem_name, paths))
            return paths[0]

        for key, list_ in cityscapes_paths.items():
            list_.append(glob_one(key))

    return cityscapes_paths

# --- Preprocessing functions/classes.
class ImagePreprocessor():
    """
    Class which allows to fetch and preprocess images in parallel.
    It preprocesses images (cropping and resizing, if necessary)
    and saves them in the `target_directory`.
    """
    def __init__(self, target_directory, crop, resize, stixel_width):
        self.target_directory = target_directory
        self.crop = crop
        self.resize = resize
        self.stixel_width = stixel_width

    def __call__(self, image_path):
        filename = os.path.basename(image_path)

        img = PIL.Image.open(image_path)
        original_shape = img.size

        resize = self.resize

        enforce_multiple_8 = True
        if enforce_multiple_8 and self.stixel_width % 8 != 0:
            raise IOError("Let's try to stick with multiples of 8 for now.")
        if self.crop is not None:
            left = self.crop[0]
            upper = self.crop[1]
            right = original_shape[0] - self.crop[0]
            lower = original_shape[1] - self.crop[1]
            img = img.crop((left, upper, right, lower))
            
            # Enforce width to be multiple of stixel width.
            if resize is None and img.size[0] % self.stixel_width:
                resize = img.size

        if resize is not None:
            # Enforce aspect ratio consistency.
            ratio = min(resize[0]/img.size[0], 
                        resize[1]/img.size[1])
            resize = np.array(img.size) * ratio

            # Enforce width to be multiple of stixel width.
            if resize[0] % self.stixel_width != 0:
                new_width = resize[0] - resize[0] % self.stixel_width
                ratio = new_width / img.size[0]
                resize = np.array(img.size) * ratio

            img = img.resize(resize.astype(np.int), PIL.Image.NEAREST)

        new_shape = img.size
        img.save(os.path.join(self.target_directory, filename))

        return original_shape, new_shape

def preprocess_images(image_dict, stixel_width, crop, resize, jobs=1):
    """
    Handles the parallel preprocessing of the images using the
    ImagePreprocessor class.
    """
    with multiprocessing.Pool(processes=jobs) as pool:
        results = [None] * len(image_dict)
        for i, (target_directory,image_paths) in enumerate(image_dict.items()):
            if not os.path.isdir(target_directory):
                print("Creating target directory: {}".format(target_directory))
                os.makedirs(target_directory)

            preprocessor = ImagePreprocessor(target_directory, crop, resize, 
                                             stixel_width)
            results[i] = pool.map_async(preprocessor, image_paths)
        pool.close()
        pool.join()

    # Join and unzip results of each map call.
    original_shapes, new_shapes =\
            zip(*sum([result.get() for result in results],[]))
    return original_shapes, new_shapes

# --- Functions for running external scripts.
def run_stixelsscript(input_directory, max_disparity, segmentationweights,
        instance_weight, disparity_weight, stixel_width, mute=False):
    """
    Call the stixel program with corresponding arguments.
    """
    if not os.path.isdir(os.path.join(input_directory,'stixels/')):
        os.makedirs(os.path.join(input_directory,'stixels/'))

    for segmentation_weight in segmentationweights:
        command = ["GPUStixels/build/stixels",
                   input_directory,
                   str(max_disparity),
                   str(segmentation_weight),
                   str(instance_weight),
                   str(disparity_weight),
                   str(stixel_width)]
        print("Running stixels: "+" ".join(command))
        if not mute:
            subprocess.check_call(command)
        else:
            subprocess.check_output(command)

def run_segmentation(input_directory, output_directory, usegtoffsets=False):
    """
    Runs the ICNet segmentation script on the files in `input_directory` and
    saves the results to `output_directory`.
    """
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    options = []
    if usegtoffsets:
        options = ["--instancegt=as_prediction"]

    #options = ["--colored"]
    model_file = "instanceoffset/weights/Net_DRNRegDs_epoch45.pth"
    model = "DRNDownsampledCombined"
    command = ["/bin/bash",
               "instanceoffset/run_segmentation.sh",
               model_file,
               input_directory,
               output_directory,
               model] + options
    print("Running segmentation: "+" ".join(command))
    subprocess.check_call(command)

def run_segmentation_ICNet(input_directory, output_directory):
    """
    Runs the ICNet segmentation script on the files in `input_directory` and
    saves the results to `output_directory`.
    """
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    command = ["/bin/bash",
               "ICNet-tensorflow/run_segmentation.sh",
               input_directory,
               output_directory]
    print("Running segmentation: "+" ".join(command))
    subprocess.check_call(command)

def run_stixelsvisualization(working_directory, max_disparity,
        visualization_config, mute=False):
    """
    Runs the stixel visualization script on the files in
    `working_directory`/stixels and saves the results to 
    `working_directory`/stixelsim.
    """
    command = ["/bin/bash",
               "clustering_visualization/run_stixelvisualization.sh",
               "-d",str(max_disparity),
               #"-t",
               os.path.abspath(working_directory)]
    if 'usegtassignment' in visualization_config.keys()\
            and visualization_config['usegtassignment']:
        command.insert(-1, "-i")
    if 'resultsonly' in visualization_config.keys()\
            and visualization_config['resultsonly']:
        command.insert(-1, "-r")

    for key in ['eps', 'min_size', 'size_filter']:
        if key in visualization_config.keys():
            command.insert(-1, "--{}".format(key))
            command.insert(-1, str(visualization_config[key]))
    print("Running stixel visualization: "+" ".join(command))
    output = subprocess.check_output(command)
    if not mute:
        print_bytes(output)
    return parse_visualization_output(output)

def print_bytes(output):
    with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as stdout:
        stdout.write(output)
        stdout.flush()

def parse_visualization_output(out_buffer):
    out_lines = str(out_buffer).split("\\n")

    stixel_lines = [line for line in out_lines[-5:]\
                    if line.startswith("Average")]

    result = re.search(' = ([0-9]*\.[0-9])', stixel_lines[0])
    if result is not None:
        avg_stixels = float(result.group(1))
        return avg_stixels

    return -1.0

def run_cityscapesevaluation_CNN(working_directory, mute=False):
    command = ["/bin/bash",
               "evaluation/run_semanticevaluation.sh",
               # The string "pred" has to be in the path for the eval
               # script. Weird.
               os.path.abspath(
                   os.path.join(working_directory, 'probs/*labelImg.png')),
               os.path.abspath(
                   os.path.join(working_directory,'gtFine/*labelIds.png'))]
    print("Running semantic CNN evaluation: "+" ".join(command))
    subprocess.check_call(command)

def run_cityscapesevaluation(working_directory, mute=False):
    """
    Runs the semantic segmentation evaluation module of Cityscapes.
    """
    command = ["/bin/bash",
               "evaluation/run_semanticevaluation.sh",
               os.path.abspath(
                   os.path.join(working_directory,'results/preds/*')),
               os.path.abspath(
                   os.path.join(working_directory,'gtFine/*labelIds.png'))]
    print("Running semantic stixel evaluation: "+" ".join(command))
    output = subprocess.check_output(command)
    if not mute:
        print_bytes(output)
    semantic_score = parse_semantic_evaluation_output(output)

    command = ["/bin/bash",
               "evaluation/run_instanceevaluation.sh",
               os.path.abspath(
                   os.path.join(working_directory,'gtFine')),
               os.path.abspath(
                   os.path.join(working_directory,'results'))]
    print("Running instance stixel evaluation: "+" ".join(command))
    output = subprocess.check_output(command)
    if not mute:
        print_bytes(output)
    instance_score = parse_instance_evaluation_output(output)

    return semantic_score, instance_score

def parse_semantic_evaluation_output(out_buffer):
    out_lines = str(out_buffer).split("\\n")

    score_lines = [line for line in out_lines[-20:]\
                   if line.startswith("Score")]

    result = re.search('[0-9]\.[0-9]{3}', score_lines[0])
    if result is not None:
        score = float(result.group(0))
        return score

    return -1.0

def parse_instance_evaluation_output(out_buffer):
    out_lines = str(out_buffer).split("\\n")

    score_lines = [line for line in out_lines[-5:]\
                   if line.startswith("average")]

    result = re.search('[0-9]\.[0-9]{3}', score_lines[0])
    if result is not None:
        score = float(result.group(0))
        return score

    return -1.0

def main(actions, directories, stixel_config, preprocess_config,
        segmentation_config, visualization_config):
    print("Running the following actions: {}".\
            format(str([a for a, v in actions.items() if v is True])))

    if actions['clean']:
        RELEVANT_DIRECTORIES = ['left', 'camera', 'disparities', 
                'probs', 'gtFine', 'stixels', 'stixelsim',
                'results']
        for dir_ in RELEVANT_DIRECTORIES:
            dir_path = os.path.abspath(
                        os.path.join(directories['working'], dir_))
            if os.path.isdir(dir_path):
                print("Removing directory: {}".format(dir_path))
                shutil.rmtree(dir_path)

    stem_names = parse_input_images(directories['files'])

    no_gt = False
    cityscapes_paths = find_cityscapes_files(CITYSCAPES_PATH, stem_names,
                                             no_gt)
    # --- Preprocess images in parallel.
    if actions['preprocess']:
        wd = lambda dir_: os.path.join(directories['working'], dir_)
        preprocess_paths = {
                wd('left/') : cityscapes_paths['left'],
                wd('disparities/') : cityscapes_paths['disparity'] }
        if not no_gt:
            preprocess_paths[wd('gtFine/')] = cityscapes_paths['semanticgt']\
                                              + cityscapes_paths['instancegt']
        original_shapes, new_shapes = preprocess_images(
                preprocess_paths,
                stixel_width=stixel_config['width'],
                crop=preprocess_config['crop'],
                resize=preprocess_config['resize'],
                jobs=preprocess_config['jobs'])

        # Checking resulting shapes.
        if not all([original_shapes[0] == shape for shape in original_shapes]):
            raise IOError("All input images must be of same size!")
        original_shape = original_shapes[0]
        if not all([new_shapes[0] == shape for shape in new_shapes]):
            raise IOError("Preprocessed images are not of same size!")
        preprocessed_shape = new_shapes[0]
        print("Original shapes were: {}".format(original_shape))
        print("New shapes are: {}".format(preprocessed_shape))
        size_factors = np.divide(preprocessed_shape, original_shape)

        # --- Copying camera files to working directory.
        camera_directory = os.path.join(directories['working'], 'camera/')
        if not os.path.isdir(camera_directory):
            os.makedirs(camera_directory)
        for camera_path in cityscapes_paths['camera']:
            if np.any(size_factors != 1):
                with open(camera_path, 'r') as camera_file:
                    camera_dict = json.load(camera_file)
                    camera_dict['original'] = copy.deepcopy(camera_dict)
                    camera_dict['intrinsic']['u0'] *= size_factors[0]
                    camera_dict['intrinsic']['v0'] *= size_factors[1]
                new_camera_path = os.path.join(camera_directory,
                                               os.path.basename(camera_path))
                with open(new_camera_path, 'w') as camera_file:
                    json.dump(camera_dict, camera_file, indent=2)
            else:
                shutil.copy(camera_path, camera_directory)

    # --- Run segmentation.
    if actions['segmentation']:
        usegtoffsets = segmentation_config['usegtoffsets']
        run_segmentation(os.path.join(directories['working'],'left/'),
                os.path.join(directories['working'],'probs/'),
                usegtoffsets)

    # --- Run stixel script for different segmentation weights.
    if actions['stixel']:
        run_stixelsscript(directories['working'], 
                stixel_config['maxdisparity'],
                stixel_config['segmentationweights'], 
                stixel_config['instanceweight'], 
                stixel_config['disparityweight'], 
                stixel_config['width'])

    # --- Run visualziation and instance clustering.
    avg_no_stixels = None
    if actions['visualization']:
        visualization_config['resultsonly'] = False
        avg_no_stixels = run_stixelsvisualization(
                directories['working'],
                stixel_config['maxdisparity'],
                visualization_config)

    semantic_score, instance_score = None, None
    if actions['evaluateCNN'] and not no_gt:
        run_cityscapesevaluation_CNN(directories['working'])
    if actions['evaluate'] and not no_gt:
        semantic_score, instance_score =\
                run_cityscapesevaluation(directories['working'])
        print("Semantic score = {}, instance score = {}, avg stixels = {}"
              .format(semantic_score, instance_score, avg_no_stixels))

def sample_config_result(directories, stixel_config, visualization_config):
    # --- Run stixel script for different segmentation weights.
    run_stixelsscript(directories['working'], 
            stixel_config['maxdisparity'],
            stixel_config['segmentationweights'], 
            stixel_config['instanceweight'], 
            stixel_config['disparityweight'], 
            stixel_config['width'],
            mute=True)

    # --- Run visualziation and instance clustering.
    avg_no_stixels = None
    visualization_config['resultsonly'] = True
    avg_no_stixels = run_stixelsvisualization(
            directories['working'],
            stixel_config['maxdisparity'],
            visualization_config,
            mute=True)

    semantic_score, instance_score =\
            run_cityscapesevaluation(directories['working'], mute=True)

    return semantic_score, instance_score, avg_no_stixels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Script to run the instance stixel pipeline.")
    # Data options, e.g. directories, resize and cropping.
    parser.add_argument(
            "FILEDIR", type=str, 
            help="Directory that contains the files that will be processed. "\
            "These only have to match the following pattern:\n"\
            "<cityscapesbasename>_[a-z].*.png\n"\
            "Thus, they can be any cityscapes png file."\
            "The corresponding required files will be search automatically.")
    parser.add_argument(
            "--workingdirectory", type=str, default=None,
            help="Directory where preprocessed and result files will be "\
            "saved. If not set, it will be the parent directory of the file "\
            "directory.")
    parser.add_argument(
            "--crop","-c", type=int, nargs=2, default=None,
            help="Specific top/bottom and left/right crop.")
    parser.add_argument(
            "--resize","-r", type=int, nargs=2, default=None,
            help="Specific final size (width,height) of images applied "\
            "after cropping.")
    parser.add_argument(
            "--jobs","-j", type=int, default=1,
            help="Maximum number of processes to spawn for preprocessing.")
    # Algorithm parameters.
    parser.add_argument(
            "--maxdisparity","-d", type=str, default=128,
            help="Maximum disparity value.")
    parser.add_argument(
            "--segmentationweights","-s", type=float, nargs='+', default=[0.0],
            help="Choose segmentation weights to process.")
    parser.add_argument(
            "--instanceweight","-i", type=float, default=0.0,
            help="Set the instance weight to process.")
    parser.add_argument(
            "--disparityweight", type=float, default=1.0,
            help="Set the instance weight to process.")
    parser.add_argument(
            "--stixelwidth","-w", type=int, default=8,
            help="Set stixel width.")
    parser.add_argument(
            "--eps", type=float, default=16,
            help="DBSCAN: eps.")
    parser.add_argument(
            "--min-size", type=int, default=2,
            help="DBSCAN: min_size.")
    parser.add_argument(
            "--size-filter", type=int, default=10,
            help="DBSCAN: size filter for core points.")
    # Boolean options.
    parser.add_argument(
            "--usegtoffsets", action='store_true', 
            help="Use ground truth offsets to compute stixels.")
    parser.add_argument(
            "--usegtassignment", action='store_true', 
            help="Use ground truth instances for assignment.")
    parser.add_argument(
            "--evaluate", action='store_true', 
            help="Run evaluation scripts.")
    parser.add_argument(
            "--nopre", action='store_true', 
            help="Skip data preprocessing.")
    parser.add_argument(
            "--nosegmentation", action='store_true', 
            help="Skip segmentation of data.")
    parser.add_argument(
            "--nostixel", action='store_true', 
            help="Skip stixel computation.")
    parser.add_argument(
            "--evaluateCNN", action='store_true', 
            help="Evaluate semantic predictions of CNN.")
    parser.add_argument(
            "--novisualization", action='store_true', 
            help="Skip visualization of data.")
    parser.add_argument(
            "--clean", action='store_true', 
            help="Remove all preprocessed data and force preprocessing.")
    parser.add_argument(
            "--nowarn", action='store_true', 
            help="Suppress all warnings.")
    args = parser.parse_args()

    if not os.path.isdir(args.FILEDIR):
        raise IOError("File directory {} not found.".format(args.FILEDIR))

    # If working directory not set, set to parent directory of file directory.
    if args.workingdirectory is None:
        args.workingdirectory = os.path.join(args.FILEDIR, os.pardir)

    if not os.path.isdir(args.workingdirectory):
        os.makedirs(args.workingdirectory)

    if args.nowarn:
        warnings.filterwarnings("ignore")

    # Package arguments in logical dictionaries.
    actions = {
            'preprocess' : not args.nopre,
            'segmentation' : not args.nosegmentation,
            'stixel' : not args.nostixel,
            'visualization' : not args.novisualization,
            'evaluate' : args.evaluate,
            'evaluateCNN' : args.evaluateCNN,
            'clean' : args.clean}
    directories = {
            'files' : args.FILEDIR,
            'working' : args.workingdirectory}
    visualization_config = {
            'usegtassignment' : args.usegtassignment,
            'eps' : args.eps,
            'min_size' : args.min_size,
            'size_filter' : args.size_filter }
    segmentation_config = {
            'usegtoffsets' : args.usegtoffsets}
    preprocess_config = {
            'crop' : args.crop,
            'resize' : args.resize,
            'jobs' : args.jobs}
    stixel_config = {
            'width' : args.stixelwidth,
            'maxdisparity' : args.maxdisparity,
            'segmentationweights' : args.segmentationweights,
            'instanceweight' : args.instanceweight,
            'disparityweight' : args.disparityweight}
    main(actions, directories, stixel_config, preprocess_config,
            segmentation_config, visualization_config)
