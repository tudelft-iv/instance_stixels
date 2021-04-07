#!/usr/bin/env bash

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

if [[ $# -lt 3 ]]
then
  echo "Usage: $0 data_directory filename segweight instweight [options]"
  exit 1
fi
DIRECTORY=$1
FILENAME=$2
SWEIGHT=$3
IWEIGHT=$4
shift; shift; shift; shift;
OPTIONS="$@"
echo "Using filename: ${FILENAME}"
echo "Using seg weight: ${SWEIGHT}"
echo "Using inst weight: ${IWEIGHT}"
echo "Using options: ${OPTIONS}"

#SFILE0="${FILENAME}w0.txt"
#rm -r data/random_subval_imgs/results/ data/random_subval_imgs/stixels data/random_subval_imgs/stixelsim
#script -c "time python3 run.py -c 132 120 data/random_subval_imgs/left_org/ -j 12 -s 0.0 --evaluate --nosegmentation" ${SFILE0}
#script -c "time python3 run.py -c 132 120 data/random_subval_imgs/left_org/ -j 12 -s 0.0 --evaluate --nopre --nosegmentation" ${SFILE0}
#grep -e 'number of stixels' -e 'Score Average' -e 'Mean deviation' ${SFILE0} | cat >> ${SFILE0}

SFILE1="${FILENAME}iw${IWEIGHT}sw${SWEIGHT}w1.txt"
rm -r ${DIRECTORY}/results/ ${DIRECTORY}/stixels ${DIRECTORY}/stixelsim
script -c "time python3 run.py -c 128 120 ${DIRECTORY}/left_org/ -j 12 -i ${IWEIGHT} -s ${SWEIGHT} --evaluate ${OPTIONS}" ${SFILE1}
grep -e 'number of stixels' -e 'Score Average' -e 'average' -e 'Mean deviation' ${SFILE1} | cat >> ${SFILE1}

#tail ${SFILE0}
tail ${SFILE1}
