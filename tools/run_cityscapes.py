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

# Description:
# This python script will automatically collect the required files, run the
# single components of the instance stixel pipeline.

import os
import sys
import shutil
import glob
import warnings

import subprocess
import multiprocessing
from copy import deepcopy

import re
import time
import json
import argparse # TODO: maybe use configargparse?

import numpy as np
import PIL.Image
try:
    import skopt
except ImportError:
    skopt = None

#import ipdb

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
START_TIME = time.strftime('%m%d-%H%M')

CITYSCAPES_PATH = "/data/Cityscapes"
if "CITYSCAPES_DATASET" in os.environ:
    CITYSCAPES_PATH = os.environ['CITYSCAPES_DATASET']

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
        # TODO: maybe do size checks here. Only possible, if they do not depend
        # on input.
        # Also, make sure that ground truth and input are transformed in the
        # same way. As long as the input size is the same, it should be fine
        # for now.
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
        instance_weight, disparity_weight, pairwise, stixel_width,
        visualization_config, use_tensorrt=False, mute=False):
    """
    Call the stixel program with corresponding arguments.
    """
    global PROJECT_DIR
    if not os.path.isdir(os.path.join(input_directory,'stixels/')):
        os.makedirs(os.path.join(input_directory,'stixels/'))

    print("use_tensorrt =", use_tensorrt)
    for segmentation_weight in segmentationweights:
        command = [os.path.join(PROJECT_DIR,"build/run_cityscapes"),
                   input_directory,
                   str(max_disparity),
                   str(segmentation_weight),
                   str(instance_weight),
                   str(disparity_weight),
                   str(pairwise),
                   str(stixel_width),
                   str(visualization_config['eps']),
                   str(visualization_config['min_size']),
                   str(visualization_config['size_filter']),
                   str(int(use_tensorrt))]
        print("Running stixels: "+" ".join(command))
        output = subprocess.check_output(command)
        if not mute:
            print_bytes(output)
    return parse_stixel_output(output)

def run_segmentation(input_directory, output_directory, segmentation_config,
        mute=False):
    """
    Runs the ICNet segmentation script on the files in `input_directory` and
    saves the results to `output_directory`.
    """
    global SCRIPT_DIR
    usegtoffsets = segmentation_config['usegtoffsets']
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    model_file = segmentation_config['modelfile']
    model = segmentation_config['modelname']
    drn_model = model[-8:] if 'drn_d_' in model else None
    model = model[:-8] if 'drn_d_' in model else model
    # ---
    #drn_model = None
    #model_file = "weights/DRNDSDoubleSegSL_0.0001_0.0001_0_0_0140.pth"
    #model = "DRNDSDoubleSegSL"
    # ---
    # This is the one used for testing.
    #model_file = "weights/Net_DRNRegDs_epoch45.pth"
    #model = "DRNDownsampledCombined"
    # --- Including depth.
    # Single net UV.
    #~model_file = "weights/DRNDSOffsetDisparity_0055.pth"
    #~model = "DRNDSOffsetDisparity"
    # Single net UVD.
    #model_file = "weights/DRNDSOffsetDisparityASL_0095.pth"
    #model = "DRNDSOffsetDisparityASL"
    #drn_model = 'drn_d_38'

    options = []
    #options = ["--colored"]
    if usegtoffsets:
        options = ["--instancegt=as_prediction"]

    if drn_model is not None:
        options = ["--drn-name",drn_model]

    command = ["/bin/bash",
               os.path.join(SCRIPT_DIR, "CNN_training/run_segmentation.sh"),
               model_file,
               input_directory,
               output_directory,
               model] + options
    print("Running segmentation: "+" ".join(command))

    output = subprocess.check_output(command)
    if not mute:
        print_bytes(output)
    return parse_segmentation_output(output)

def run_stixelsvisualization(working_directory, max_disparity,
        visualization_config, mute=False):
    """
    Runs the stixel visualization script on the files in
    `working_directory`/stixels and saves the results to
    `working_directory`/stixelsim.
    """
    global SCRIPT_DIR
    command = ["/bin/bash",
               os.path.join(SCRIPT_DIR,
                            "visualization/run_stixelvisualization.sh"),
               "-d", str(max_disparity),
               "-t",
               os.path.abspath(working_directory)]
    if 'usegtassignment' in visualization_config.keys()\
            and visualization_config['usegtassignment']:
        command.insert(-1, "-i")
    if 'resultsonly' in visualization_config.keys()\
            and visualization_config['resultsonly']:
        command.insert(-1, "-r")
    if visualization_config['use-disparity'] != "":
        command.insert(-1, "--use-disparity")
        command.insert(-1, visualization_config['use-disparity'])

    for key in ['eps', 'min_size', 'size_filter']:
        if key in visualization_config.keys():
            command.insert(-1, "--{}".format(key))
            command.insert(-1, str(visualization_config[key]))
    print("Running stixel visualization: "+" ".join(command))
    output = subprocess.check_output(command)
    if not mute:
        print_bytes(output)
    #sys.stdout.buffer.write(output)
    return parse_visualization_output(output)

def print_bytes(output):
    with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as stdout:
        stdout.write(output)
        stdout.flush()

def parse_stixel_output(out_buffer):
    out_lines = str(out_buffer).split("\\n")

    time_line = [line for line in out_lines[-3:]\
                 if line.startswith("It took an average")][0]

    result = re.search('([0-9]*\.[0-9]*) milliseconds', time_line)
    if result is not None:
        average_time = float(result.group(1))
        return average_time*1e-3

    return -1.0

def parse_segmentation_output(out_buffer):
    out_lines = str(out_buffer).split("\\n")

    time_line = [line for line in out_lines[-3:]\
                 if line.startswith("Time per batch")][0]

    result = re.search('([0-9]*\.[0-9]*) s', time_line)
    if result is not None:
        time_per_batch = float(result.group(1))
        return time_per_batch

    return -1.0

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
    global SCRIPT_DIR
    command = ["/bin/bash",
               os.path.join(SCRIPT_DIR,
                            "evaluation/run_semanticevaluation.sh"),
               # The string "pred" has to be in the path for the eval
               # script.
               os.path.abspath(
                   os.path.join(working_directory, 'probs/*labelImg.png')),
               os.path.abspath(
                   os.path.join(working_directory,'gtFine/*labelIds.png'))]
    print("Running semantic CNN evaluation: "+" ".join(command))
    subprocess.check_call(command)

def run_cityscapesevaluation(working_directory, mute=False):
    # TODO: Write a python script which uses the cityscapes modules to evaluate
    # the predictions.
    """
    Runs the semantic segmentation evaluation module of Cityscapes.
    """
    global SCRIPT_DIR
    command_semantic = [
            "/bin/bash",
            os.path.join(SCRIPT_DIR,
                         "evaluation/run_semanticevaluation.sh"),
            os.path.abspath(
                os.path.join(working_directory,'results/preds/*')),
            os.path.abspath(
                os.path.join(working_directory,'gtFine/*labelIds.png'))]
    print("Running semantic stixel evaluation: "+" ".join(command_semantic))
    command_instance = [
            "/bin/bash",
            os.path.join(SCRIPT_DIR,
                         "evaluation/run_instanceevaluation.sh"),
            os.path.abspath(
                os.path.join(working_directory,'gtFine')),
            os.path.abspath(
                os.path.join(working_directory,'results'))]
    print("Running instance stixel evaluation: "+" ".join(command_instance))

    commands = [command_semantic, command_instance]
    with multiprocessing.Pool(processes=2) as pool:
        outputs = pool.map(subprocess.check_output, commands)
        #output = subprocess.check_output(command)
    #sys.stdout.buffer.write(output)
    if not mute:
        for output in outputs:
            print_bytes(output)

    semantic_score = parse_semantic_evaluation_output(outputs[0])
    instance_score = parse_instance_evaluation_output(outputs[1])

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

    if len(score_lines) > 0:
        result = re.search('[0-9]\.[0-9]{3}', score_lines[0])
        if result is not None:
            score = float(result.group(0))
            return score

    return -1.0

def main(actions, directories, stixel_config, preprocess_config,
        segmentation_config, visualization_config):
    # TODO: use abspaths
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

    # --- Preprocess images in parallel.
    stem_names =  [re.sub('_[a-z].*','',os.path.basename(f))\
                   for f in os.listdir(directories['files'])]

    # TODO: cityscapes directory is a good candidate for a config file.
    no_gt = False
    cityscapes_paths = find_cityscapes_files(CITYSCAPES_PATH, stem_names,
                                             no_gt)
    #print("Cityscapes paths: {}".format(cityscapes_paths))
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
        #if size_factor[0] != size_factor[1]:
        #    raise IOError("Aspect ratio of preprocessed images is inconsistent: "\
        #            "Size factor: {}".format(size_factor))
        #size_factor = size_factor[0]

        # --- Copying camera files to working directory.
        # TODO: modify camera files on the fly!
        camera_directory = os.path.join(directories['working'], 'camera/')
        if not os.path.isdir(camera_directory):
            os.makedirs(camera_directory)
        for camera_path in cityscapes_paths['camera']:
            if np.any(size_factors != 1):
                with open(camera_path, 'r') as camera_file:
                    camera_dict = json.load(camera_file)
                    camera_dict['original'] = deepcopy(camera_dict)
                    camera_dict['intrinsic']['u0'] *= size_factors[0]
                    camera_dict['intrinsic']['v0'] *= size_factors[1]
                new_camera_path = os.path.join(camera_directory,
                                               os.path.basename(camera_path))
                with open(new_camera_path, 'w') as camera_file:
                    json.dump(camera_dict, camera_file, indent=2)
            else:
                shutil.copy(camera_path, camera_directory)

    # --- Run segmentation.
    CNN_time_per_batch = -1
    if actions['segmentation']:
        if not stixel_config['tensorrt']:
            CNN_time_per_batch = run_segmentation(
                    os.path.join(directories['working'],'left/'),
                    os.path.join(directories['working'],'probs/'),
                    segmentation_config)
    # --- Run stixel script for different segmentation weights.
    stixel_time_per_frame = -1
    if actions['stixel']:
        if (stixel_config['tensorrt'] and
                "DRNDSDoubleSegSL_0.0001_0.0001_0_0_0095"
                not in segmentation_config['modelfile']):
            raise NotImplementedError(
                    "TensorRT inference is only supported for "
                    "network "
                    "DRNDSDoubleSegSL_0.0001_0.0001_0_0_0095 "
                    "at the moment.")

        stixel_time_per_frame = run_stixelsscript(
                directories['working'],
                stixel_config['maxdisparity'],
                stixel_config['segmentationweights'],
                stixel_config['instanceweight'],
                stixel_config['disparityweight'],
                stixel_config['pairwise'],
                stixel_config['width'],
                visualization_config,
                stixel_config['tensorrt'])

    # --- Run visualziation and instance clustering.
    avg_no_stixels = None
    if actions['visualization']:
        visualization_config['resultsonly'] = False
        avg_no_stixels = run_stixelsvisualization(
                directories['working'],
                stixel_config['maxdisparity'],
                visualization_config)

    semantic_score, instance_score = None, None
    if actions['evaluateCNN']:
        run_cityscapesevaluation_CNN(directories['working'])
    if actions['evaluate']:
        semantic_score, instance_score =\
                run_cityscapesevaluation(directories['working'])
        #run_disparityevaluation(directories['working'])
    print("CNN time per frame {} s, {} fps"
          .format(CNN_time_per_batch, 1./CNN_time_per_batch))
    print("Stixel time per frame {} s, {} fps"
          .format(stixel_time_per_frame, 1./stixel_time_per_frame))
    total_time_per_frame = CNN_time_per_batch + stixel_time_per_frame
    print("Total processing time per frame {} s, fps {}"
          .format(total_time_per_frame, 1./total_time_per_frame))
    print("Semantic score = {}, instance score = {}, avg stixels = {}"
          .format(semantic_score, instance_score, avg_no_stixels))

    optimize_params = 0 #True
    if not optimize_params:
        return 0
    print("\n\n\n------------ Starting parameter optimization.")

    optimize_instance_weight = (stixel_config['instanceweight'] != 0.0)

    # --- Sample function.
    config_results = {
            'arguments' : {
                "actions"              : deepcopy(actions),
                "directories"          : deepcopy(directories),
                "stixel_config"        : deepcopy(stixel_config),
                "preprocess_config"    : deepcopy(preprocess_config),
                "segmentation_config"  : deepcopy(segmentation_config),
                "visualization_config" : deepcopy(visualization_config) },
            'results' : [[0]*5],
            'best_score' : 0 }
    best_scores = [0] * 4
    def sample_result(params):
        #segmentationweight, disparityweight = params[:2]
        #eps, min_size, size_filter = params
        if optimize_instance_weight:
            segmentationweight, instanceweight, disparityweight = params[:3]
            eps, min_size, size_filter = params[3:]
        else:
            segmentationweight, disparityweight = params[:2]
            eps, min_size, size_filter = params[2:]
            instanceweight = 0
        # --- Remove previous result directories.
        shutil.rmtree(os.path.join(directories['working'], 'results/'),
                      ignore_errors=True)
        shutil.rmtree(os.path.join(directories['working'], 'stixels/'),
                      ignore_errors=True)
        shutil.rmtree(os.path.join(directories['working'], 'stixelsim/'),
                      ignore_errors=True)

        # --- set parameters for current run
        stixel_config['segmentationweights'] = [segmentationweight]
        stixel_config['instanceweight'] = instanceweight
        stixel_config['disparityweight'] = disparityweight
        visualization_config['eps'] = eps
        visualization_config['min_size'] = min_size
        visualization_config['size_filter'] = size_filter

        semantic_score, instance_score, avg_no_stixels =\
                sample_config_result(
                    directories,
                    stixel_config,
                    visualization_config)

        score = semantic_score + 1.5 * instance_score

        if score > best_scores[0]:
            best_scores[0] = score
            best_scores[1] = semantic_score
            best_scores[2] = instance_score
            best_scores[3] = avg_no_stixels

        run_id = config_results['results'][-1][0] + 1
        config_result = [run_id,
                         score, semantic_score, instance_score, avg_no_stixels,
                         float(stixel_config['segmentationweights'][0]),
                         float(stixel_config['instanceweight']),
                         float(stixel_config['disparityweight']),
                         float(visualization_config['eps']),
                         int(visualization_config['min_size']),
                         int(visualization_config['size_filter'])]
        config_results['results'].append(config_result)
        config_results['best_score'] = best_scores
        # write config results
        json_filename = "config_results{}.json".format(START_TIME)
        with open(json_filename, 'w') as json_file:
            json.dump(config_results, json_file, indent=1)

        print("config_result = {}".format(config_result))
        print("best_scores = {}".format(best_scores))

        return -score # minimize

    #x0 = [3.876,10.0,0.0001,38.7,2,34] # old, u,v default
    #x0  = [0.5,0.0004,0.25,1.0,2,30] # 3d default
    x0  = [4.709500548254913, 0.0031312903639774976, 0.0001,
           18.82232269133926, 3, 25]
    #x0 = [0.8446,10.0,0.0001,27.1,3,23]
    #x0 = [1.0,1.0,1.0,16.0,2,10]
    #x0 = [16.0,2,10]

    # Only if you want to optimize hyperparameters.
    if skopt is not None:
        space = [skopt.space.Real(
                     10**-4, 10**2, "log-uniform", name='segmentationweight'),
                 skopt.space.Real(
                     10**-4, 10**1, "log-uniform", name='disparityweight'),
                 skopt.space.Real(
                     1.0, 100.0, "uniform", name='eps'),
                     #0.1, 10.0, "uniform", name='eps'),
                 skopt.space.Integer(1, 10, name='min_size'),
                 skopt.space.Integer(1, 100, name='size_filter')]
        if optimize_instance_weight:
            space.insert(1,
                         skopt.space.Real(10**-4, 10**2,
                                          "log-uniform",
                                          name='instanceweight'))
        elif len(x0) > 5:
            del x0[1]


        res_gp = skopt.gp_minimize(
                        sample_result,
                        space,
                        acq_func='PI',
                        n_random_starts=100,
                        n_calls=500,
                        x0=x0)
    #print("If you want to plot the results, you need to change the matplotlib "
    #      "backend first.")

def sample_config_result(directories, stixel_config, visualization_config):
    # --- Run stixel script for different segmentation weights.
    try:
        run_stixelsscript(directories['working'],
                stixel_config['maxdisparity'],
                stixel_config['segmentationweights'],
                stixel_config['instanceweight'],
                stixel_config['disparityweight'],
                stixel_config['pairwise'],
                stixel_config['width'],
                visualization_config,
                stixel_config['tensorrt'],
                mute=True)
    except subprocess.CalledProcessError:
        return 0.0, 0.0, 0.0

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
            # TODO: Maybe parse dict by using type=json.loads to parse all
            # weights.
            "--segmentationweights","-s", type=float, nargs='+', default=[0.0],
            help="Choose segmentation weights to process.")
    parser.add_argument(
            "--instanceweight","-i", type=float, default=0.0,
            help="Set the instance weight to process.")
    parser.add_argument(
            "--disparityweight", type=float, default=1.0,
            help="Set the instance weight to process.")
    parser.add_argument(
            "--pairwise", type=int, default=1, choices=(0,1),
            help="Whether to use unary or pairwise (default) regularization (0 or 1).")
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
            "--use-disparity", type=str, default="",
            help="Use median disparity for clustering. Either 'from_pred' or "
            "'from_gt'. Note: So far only affects clustering script.")
    parser.add_argument(
            "--modelfile", type=str,
            #default="weights/Net_DRNRegDs_epoch45.pth",
            help="File path where model weights are stored.")
    parser.add_argument(
            "--modelname", type=str, default="DRNDownsampledCombined",
            help="Name of the CNN model.")
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
            "--tensorrt", action='store_true',
            help="Use TensorRT for CNN inference.")
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
    # Stuff that may be useful...
    #type=str,
    #choices=['CIFAR10','MNIST'],
    #default=[2], type=int, nargs='+',
    #action='store_true',
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
            'use-disparity' : args.use_disparity,
            'eps' : args.eps,
            'min_size' : args.min_size,
            'size_filter' : args.size_filter }
    segmentation_config = {
            'usegtoffsets' : args.usegtoffsets,
            'modelfile' : args.modelfile,
            'modelname' : args.modelname }
    preprocess_config = {
            'crop' : args.crop,
            'resize' : args.resize,
            'jobs' : args.jobs}
    stixel_config = {
            'width' : args.stixelwidth,
            'maxdisparity' : args.maxdisparity,
            'segmentationweights' : args.segmentationweights,
            'instanceweight' : args.instanceweight,
            'disparityweight' : args.disparityweight,
            'tensorrt' : args.tensorrt,
            'pairwise' : args.pairwise}
    main(actions, directories, stixel_config, preprocess_config,
            segmentation_config, visualization_config)
