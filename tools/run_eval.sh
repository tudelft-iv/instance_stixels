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

# compute mean in bash (one-liner):
# FILES="eval_logs/IS+drn_d_38+no_prior_?.txt"; FILES=("${FILES}"); echo "$(grep '^Total processing time per frame' ${FILES[@]} | sed -e 's/.* frame \(.*\) s.*/\1/' | xargs -d'\n' | sed -e 's/ / + /g' | bc -l) / ${#FILES[@]}.0" | bc -l
# compute mean and std deviation in bash/python (one-liner):
# FILES="eval_logs/IS+drn_d_38+no_prior_?.txt"; FILES=("${FILES}"); python3 -c "import numpy as np; numbers=np.array([$(grep '^Total processing time' ${FILES[@]} | sed -e 's/.* frame \(.*\) s.*/\1/' | xargs -d'\n' | sed -e 's/ /,/g')]); print(numbers.mean(), numbers.std())"

# Save all of the into table.txt
# for BASENAME in "SS+drn_d_22" "IS+drn_d_22" "SS+drn_d_38" "IS+drn_d_38"; do BASENAME="${BASENAME}+no_prior"; FILES="eval_logs/${BASENAME}*"; FILES=(${FILES}); python3 -c "import numpy as np; numbers=np.array([$(grep '^Total processing time per frame' ${FILES[@]} | sed -e 's/.* frame \(.*\) s.*/\1/' | xargs -d'\n' | sed -e 's/ /,/g')]); print(numbers.mean(), numbers.std())" | xargs echo "${BASENAME} $@" >> table.txt; done;

BASE_COMMAND="python3 run.py -c 128 120 -j 12 -w 8 --clean --evaluate"
DATA_DIR="~/Code/data/instance_stixels/cityscapes_val/left_org/"
#DATA_DIR="~/Code/data/instance_stixels/instance_argument_subval/left_org/"

source activate instance_stixels
for IDX in `seq 1 3`;
do
  MODEL_NAME="--modelname DRNDSDoubleSegSLdrn_d_22"

  NAME="IS+drn_d_22"
  PARAMS="-i 0.0009183590453334048 --disparityweight 0.00031498839514148533 -s 2.55368182480918 --eps 15.41794926189713 --min-size 3 --size-filter 1 --modelfile weights/drn_d_22/DRNDSDoubleSegSL_1e-05_0.0001_0_0_0065.pth"
  script -c "${BASE_COMMAND} ${MODEL_NAME} ${PARAMS} ${DATA_DIR}" eval_logs/${NAME}_${IDX}.txt

  NAME="SS+drn_d_22"
  PARAMS="-i 0 --disparityweight 10.0 -s 7.014718166422807 --eps 19.151487025340316 --min-size 1 --size-filter 32 --modelfile weights/drn_d_22/DRNDSDoubleSegSL_1e-05_0.0001_0_0_0065.pth"
  script -c "${BASE_COMMAND} ${MODEL_NAME} ${PARAMS} ${DATA_DIR}" eval_logs/${NAME}_${IDX}.txt

  MODEL_NAME="--modelname DRNDSDoubleSegSLdrn_d_38"

  NAME="IS+drn_d_38"
  PARAMS="-i 0.0031312903639774976 --disparityweight 0.0001 -s 4.709500548254913 --eps 18.82232269133926 --min-size 3 --size-filter 25 --modelfile weights/drn_d_38/DRNDSDoubleSegSL_0.0001_0.0001_0_0_0095.pth"
  script -c "${BASE_COMMAND} ${MODEL_NAME} ${PARAMS} ${DATA_DIR}" eval_logs/${NAME}_${IDX}.txt

  NAME="SS+drn_d_38"
  PARAMS="-i 0 --disparityweight 0.004332261642971551 -s 0.3129984051327578 --eps 19.612395959136297 --min-size 3 --size-filter 28 --modelfile weights/drn_d_38/DRNDSDoubleSegSL_0.0001_0.0001_0_0_0095.pth"
  script -c "${BASE_COMMAND} ${MODEL_NAME} ${PARAMS} ${DATA_DIR}" eval_logs/${NAME}_${IDX}.txt

done
source deactivate
