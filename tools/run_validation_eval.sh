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

if [[ $# -lt 1 ]]
then
  echo "Usage: $0 data_directory"
  exit 1
fi
DIRECTORY=$1
OUTDIR="valscript"

mkdir -p ${DIRECTORY}/${OUTDIR}

FILES=()

# no gt
SWEIGHT="0.44162"
IWEIGHT="0.00038"
DWEIGHT="0.24045"
EPS="75.06"
MIN_SIZE="1"
SIZE_FILTER="57"
OPTIONS="--evaluateCNN --clean"
SFILE1="${OUTDIR}/i${IWEIGHT}s${SWEIGHT}d${DWEIGHT}_e${EPS}ms${MIN_SIZE}sf${SIZE_FILTER}_lo_cas"
script -c "time python3 run.py -c 128 120 ${DIRECTORY}/left_org/ -j 12 -i ${IWEIGHT} -s ${SWEIGHT} --disparityweight ${DWEIGHT} --eps ${EPS} --min-size ${MIN_SIZE} --size-filter ${SIZE_FILTER} --evaluate ${OPTIONS}" ${DIRECTORY}/${SFILE1}.txt
#grep -e 'number of stixels' -e 'Score Average' -e 'average' -e 'Mean deviation' ${DIRECTORY}/${SFILE1}.txt | cat >> ${DIRECTORY}/${SFILE1}.txt
mkdir -p ${DIRECTORY}/${SFILE1}
mv ${DIRECTORY}/results/ ${DIRECTORY}/stixelsim ${DIRECTORY}/${SFILE1}

FILES+=(${SFILE1})

# gt as
SWEIGHT="0.44162"
IWEIGHT="0.00038"
DWEIGHT="0.24045"
EPS="75.06"
MIN_SIZE="1"
SIZE_FILTER="57"
OPTIONS="--usegtassignment --nopre --nosegmentation --nostixel"
SFILE1="${OUTDIR}/i${IWEIGHT}s${SWEIGHT}d${DWEIGHT}_e${EPS}ms${MIN_SIZE}sf${SIZE_FILTER}_lo_gtas"
script -c "time python3 run.py -c 128 120 ${DIRECTORY}/left_org/ -j 12 -i ${IWEIGHT} -s ${SWEIGHT} --disparityweight ${DWEIGHT} --evaluate --eps ${EPS} --min-size ${MIN_SIZE} --size-filter ${SIZE_FILTER} ${OPTIONS}" ${DIRECTORY}/${SFILE1}.txt
#grep -e 'number of stixels' -e 'Score Average' -e 'average' -e 'Mean deviation' ${DIRECTORY}/${SFILE1}.txt | cat >> ${DIRECTORY}/${SFILE1}.txt
mkdir -p ${DIRECTORY}/${SFILE1}
mv ${DIRECTORY}/results/ ${DIRECTORY}/stixelsim ${DIRECTORY}/${SFILE1}

FILES+=(${SFILE1})

# semstix no gt
SWEIGHT="10.0"
IWEIGHT="0.0"
DWEIGHT="0.0001"
EPS="74.1946"
MIN_SIZE="2"
SIZE_FILTER="49"
OPTIONS="--clean"
SFILE1="${OUTDIR}/i${IWEIGHT}s${SWEIGHT}d${DWEIGHT}_e${EPS}ms${MIN_SIZE}sf${SIZE_FILTER}_lo_cas"
script -c "time python3 run.py -c 128 120 ${DIRECTORY}/left_org/ -j 12 -i ${IWEIGHT} -s ${SWEIGHT} --disparityweight ${DWEIGHT} --evaluate --eps ${EPS} --min-size ${MIN_SIZE} --size-filter ${SIZE_FILTER} ${OPTIONS}" ${DIRECTORY}/${SFILE1}.txt
#grep -e 'number of stixels' -e 'Score Average' -e 'average' -e 'Mean deviation' ${DIRECTORY}/${SFILE1}.txt | cat >> ${DIRECTORY}/${SFILE1}.txt
mkdir -p ${DIRECTORY}/${SFILE1}
mv ${DIRECTORY}/results/ ${DIRECTORY}/stixelsim ${DIRECTORY}/${SFILE1}

FILES+=(${SFILE1})

# semstix gt assignment
SWEIGHT="10.0"
IWEIGHT="0.0"
DWEIGHT="0.0001"
EPS="74.1946"
MIN_SIZE="2"
SIZE_FILTER="49"
OPTIONS="--usegtassignment --nopre --nosegmentation --nostixel"
SFILE1="${OUTDIR}/i${IWEIGHT}s${SWEIGHT}d${DWEIGHT}_e${EPS}ms${MIN_SIZE}sf${SIZE_FILTER}_lo_gtas"
script -c "time python3 run.py -c 128 120 ${DIRECTORY}/left_org/ -j 12 -i ${IWEIGHT} -s ${SWEIGHT} --disparityweight ${DWEIGHT} --evaluate --eps ${EPS} --min-size ${MIN_SIZE} --size-filter ${SIZE_FILTER} ${OPTIONS}" ${DIRECTORY}/${SFILE1}.txt
#grep -e 'number of stixels' -e 'Score Average' -e 'average' -e 'Mean deviation' ${DIRECTORY}/${SFILE1}.txt | cat >> ${DIRECTORY}/${SFILE1}.txt
mkdir -p ${DIRECTORY}/${SFILE1}
mv ${DIRECTORY}/results/ ${DIRECTORY}/stixelsim ${DIRECTORY}/${SFILE1}

FILES+=(${SFILE1})

# gt offsets
SWEIGHT="1.0"
IWEIGHT="1.0"
DWEIGHT="1.0"
EPS="16.0"
MIN_SIZE="2"
SIZE_FILTER="10"
OPTIONS="--usegtoffsets --clean"
SFILE1="${OUTDIR}/i${IWEIGHT}s${SWEIGHT}d${DWEIGHT}_e${EPS}ms${MIN_SIZE}sf${SIZE_FILTER}_gto_cas"
script -c "time python3 run.py -c 128 120 ${DIRECTORY}/left_org/ -j 12 -i ${IWEIGHT} -s ${SWEIGHT} --disparityweight ${DWEIGHT} --evaluate --eps ${EPS} --min-size ${MIN_SIZE} --size-filter ${SIZE_FILTER} ${OPTIONS}" ${DIRECTORY}/${SFILE1}.txt
#grep -e 'number of stixels' -e 'Score Average' -e 'average' -e 'Mean deviation' ${DIRECTORY}/${SFILE1}.txt | cat >> ${DIRECTORY}/${SFILE1}.txt
mkdir -p ${DIRECTORY}/${SFILE1}
mv ${DIRECTORY}/results/ ${DIRECTORY}/stixelsim ${DIRECTORY}/${SFILE1}

FILES+=(${SFILE1})

# gt offsets + assignment
SWEIGHT="1.0"
IWEIGHT="1.0"
DWEIGHT="1.0"
EPS="16.0"
MIN_SIZE="2"
SIZE_FILTER="10"
OPTIONS="--usegtoffsets --usegtassignment --nopre --nosegmentation --nostixel"
SFILE1="${OUTDIR}/i${IWEIGHT}s${SWEIGHT}d${DWEIGHT}_e${EPS}ms${MIN_SIZE}sf${SIZE_FILTER}_gto_gtas"
script -c "time python3 run.py -c 128 120 ${DIRECTORY}/left_org/ -j 12 -i ${IWEIGHT} -s ${SWEIGHT} --disparityweight ${DWEIGHT} --evaluate --eps ${EPS} --min-size ${MIN_SIZE} --size-filter ${SIZE_FILTER} ${OPTIONS}" ${DIRECTORY}/${SFILE1}.txt
#grep -e 'number of stixels' -e 'Score Average' -e 'average' -e 'Mean deviation' ${DIRECTORY}/${SFILE1}.txt | cat >> ${DIRECTORY}/${SFILE1}.txt
mkdir -p ${DIRECTORY}/${SFILE1}
mv ${DIRECTORY}/results/ ${DIRECTORY}/stixelsim ${DIRECTORY}/${SFILE1}

FILES+=(${SFILE1})

for SFILE1 in "${FILES[@]}"
do
  echo "tail ${DIRECTORY}/${SFILE1}.txt"
  tail ${DIRECTORY}/${SFILE1}.txt
done
