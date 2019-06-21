#!/usr/bin/env bash

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

echo -e "\n--- Running semantic segmentation."
#cd ICNet-tensorflow
#cd $(dirname $0)
MODELFILE=$1
INPUT_DIR=$2
OUT_DIR=$3
MODELNAME=$4
shift; shift; shift; shift;
OPTIONS="$@"
source activate instance_stixels
echo "Executing:"\
  "python $(dirname $0)/inference.py ${MODELFILE} ${INPUT_DIR} "\
  "--save-dir=${OUT_DIR} --model=${MODELNAME} ${OPTIONS}"
python $(dirname $0)/inference.py ${MODELFILE} ${INPUT_DIR}\
  --save-dir=${OUT_DIR} --model=${MODELNAME} ${OPTIONS}
#python inference.py
#  tmp/train/models/Net_epoch260.pth
#  tmp/inference/instance_argument_subval/
#  --save-dir=tmp/inference/instance_argument_subval_out/ --model=DRNDoubleSeg
source deactivate
cd ..

