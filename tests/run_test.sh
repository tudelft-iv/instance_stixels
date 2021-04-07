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

PROJECT_DIRECTORY="$(realpath "$0" | xargs dirname)/.."
TEST_DIRECTORY="${PROJECT_DIRECTORY}/tests"
if [[ -z ${DATA_DIRECTORY+x} ]]
then
  DATA_DIRECTORY="${TEST_DIRECTORY}/data"
fi
WEIGHT_DIRECTORY="${PROJECT_DIRECTORY}/weights/"
RUN_SCRIPT="${PROJECT_DIRECTORY}/tools/run_cityscapes.py"

echo $0
# Generate file(names) for long test
for FILE in `cat ${TEST_DIRECTORY}/long_test_files.txt`
do
  mkdir -p ${DATA_DIRECTORY}/long_test/left_org
  touch ${DATA_DIRECTORY}/long_test/left_org/${FILE}
done
# Generate file(names) for short test
for FILE in `cat ${TEST_DIRECTORY}/short_test_files.txt`
do
  mkdir -p ${DATA_DIRECTORY}/short_test/left_org
  touch ${DATA_DIRECTORY}/short_test/left_org/${FILE}
done

function surpress_output(){
  while read data
  do
    if [[ "$1" == "true" ]]
    then
      echo "${data}"
    fi
  done
}

VERBOSE="false"
if [[ "$@" =~ "verbose" ]]
then
  VERBOSE="true"
fi

REGULARIZER="pairwise"
if [[ "$@" =~ "unary" ]]
then
    REGULARIZER="unary"
fi

CITYSCAPES="true"
if [[ "${CITYSCAPES}" == "true" ]]
then
  # source conda environment
  if ! source activate instance_stixels
  then
    echo "Error: Please create the \"instance_stixels\" conda environment "\
         "using the file tools/instance_stixel_env.yml."
    exit 1
  fi

  echo "Regularizer is: ${REGULARIZER}"
  if [[ "${REGULARIZER}" == "pairwise" ]]
  then
      if [[ "$1" =~ "long" ]]
      then
        TESTCASE="Long"
        echo "Starting long tests. It should take about 6 minutes."
        LOG_FILE="${DATA_DIRECTORY}/tmp_long.log"
        script -c\
          "python3 ${RUN_SCRIPT} -c 128 120 -j 12 -w 8\
           -i 0.0031312903639774976 --disparityweight 0.0001\
           -s 4.709500548254913 --eps 18.82232269133926 --min-size 3\
           --size-filter 25 --pairwise 1 --evaluate --clean\
           --modelfile ${WEIGHT_DIRECTORY}/drn_d_38/DRNDSDoubleSegSL_0.0001_0.0001_0_0_0095.pth\
           --modelname DRNDSDoubleSegSLdrn_d_38\
           ${DATA_DIRECTORY}/long_test/left_org/" ${LOG_FILE}\
           | surpress_output ${VERBOSE}
        TEST_VALUES="$(awk '/^Semantic score/ {print 1.0*$4 " " 1.0*$8 " " 1.0*$12}' ${LOG_FILE})"
        TRUE_VALUES="0.664 0.158 1421.1"
      else
        TESTCASE="Short"
        echo "Starting short tests. It should take about 16 seconds."
        LOG_FILE="${DATA_DIRECTORY}/tmp_short.log"
        script -c\
          "python3 ${RUN_SCRIPT} -c 128 120 -j 12 -w 8 -i 0.00038 --disparityweight 0.24045\
          -s 0.44162 --eps 75.06 --min-size 1 --size-filter 57 --pairwise 1 --evaluate --clean\
           --modelfile ${WEIGHT_DIRECTORY}/drn_d_38/DRNDSDoubleSegSL_0.0001_0.0001_0_0_0095.pth\
           --modelname DRNDSDoubleSegSLdrn_d_38\
           ${DATA_DIRECTORY}/short_test/left_org/" ${LOG_FILE}\
           | surpress_output ${VERBOSE}
        TEST_VALUES="$(awk '/^Semantic score/ {print 1.0*$4 " " 1.0*$8 " " 1.0*$12}' ${LOG_FILE})"
        TRUE_VALUES="0.511 0.088 1290.5" # using inference.py
      fi
  else # if [[ "${REGULARIZER}" =~ "unary" ]]
      if [[ "$1" =~ "long" ]]
      then
        TESTCASE="Long"
        echo "Starting long tests. It should take about 6 minutes."
        LOG_FILE="${DATA_DIRECTORY}/tmp_long.log"
        script -c\
          "python3 ${RUN_SCRIPT} -c 128 120 -j 12 -w 8\
           -i 0.013686917379717443 --disparityweight 0.0006375354572396317\
           -s 14.94984454762259 --eps 18.54 --min-size 4 --size-filter 35\
           --pairwise 0\
           --modelfile ${WEIGHT_DIRECTORY}/drn_d_38/DRNDSDoubleSegSL_0.0001_0.0001_0_0_0095.pth\
           --modelname DRNDSDoubleSegSLdrn_d_38\
           --evaluate --clean ${DATA_DIRECTORY}/long_test/left_org/" ${LOG_FILE}\
           | surpress_output ${VERBOSE}
        TEST_VALUES="$(awk '/^Semantic score/ {print 1.0*$4 " " 1.0*$8 " " 1.0*$12}' ${LOG_FILE})"
        TRUE_VALUES="0.669 0.163 2277.6"
      else
        TESTCASE="Short"
        echo "Starting short tests. It should take about 16 seconds."
        LOG_FILE="${DATA_DIRECTORY}/tmp_short.log"
        script -c\
          "python3 ${RUN_SCRIPT} -c 128 120 -j 12 -w 8\
           -i 0.013686917379717443 --disparityweight 0.0006375354572396317\
           -s 14.94984454762259 --eps 18.54 --min-size 4 --size-filter 35\
           --pairwise 0\
           --modelfile ${WEIGHT_DIRECTORY}/drn_d_38/DRNDSDoubleSegSL_0.0001_0.0001_0_0_0095.pth\
           --modelname DRNDSDoubleSegSLdrn_d_38\
           --evaluate --clean ${DATA_DIRECTORY}/short_test/left_org/" ${LOG_FILE}\
           | surpress_output ${VERBOSE}
        TEST_VALUES="$(awk '/^Semantic score/ {print 1.0*$4 " " 1.0*$8 " " 1.0*$12}' ${LOG_FILE})"
        TRUE_VALUES="0.52 0.124 1947.4" # using inference.py
      fi
  fi
fi

echo "Checking:"
echo -e "test_values\t\t== true_values:"
echo -e "${TEST_VALUES}\t== ${TRUE_VALUES}"
echo ""

RED="\033[0;31m"
GREEN="\033[0;32m"
NC="\033[0;0m"
if [[ "${TEST_VALUES}" == "${TRUE_VALUES}" ]]
then
  echo -e "${GREEN}${TESTCASE} test passed!${NC}"
else
  echo -e "${RED}${TESTCASE} test failed!${NC}"
fi

echo "Run \"rm -r ${DATA_DIRECTORY}/long_test ${DATA_DIRECTORY}/short_test\"?"
read -p "Run the above command to remove the data directories? (y/n)"\
     -n 1 -r -- REMOVE_DATA
echo
if [[ ${REMOVE_DATA} =~ ^[Yy]$ ]]
then
  echo "Removing data directories."
  rm -r ${DATA_DIRECTORY}/long_test ${DATA_DIRECTORY}/short_test
fi
