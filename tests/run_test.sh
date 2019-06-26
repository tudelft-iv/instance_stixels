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

TEST_DIRECTORY=$(dirname $0)
DATA_DIRECTORY="${TEST_DIRECTORY}/data/"
WORKING_DIRECTORY=$(realpath "$0" | xargs dirname)

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

# source conda environment
source activate instance_stixels


if [[ "$1" =~ "long" ]]
then
  TESTCASE="Long"
  echo "Starting long tests. It should take about 17 minutes."
  LOG_FILE="${WORKING_DIRECTORY}/tmp_long.log"
  script -c\
    "python3 run.py -c 128 120 -j 12 -w 8 -i 0.00038 --disparityweight 0.24045\
    -s 0.44162 --eps 75.06 --min-size 1 --size-filter 57 --evaluate --clean\
     ${DATA_DIRECTORY}/long_test/left_org/" ${LOG_FILE}\
     | surpress_output ${VERBOSE}
  TEST_VALUES="$(awk '/^Semantic score/ {print 1.0*$4 " " 1.0*$8 " " 1.0*$12}' ${LOG_FILE})"
  TRUE_VALUES="0.652 0.114 1374.6"
else
  TESTCASE="Short"
  echo "Starting short tests. It should take about 28 seconds."
  LOG_FILE="${WORKING_DIRECTORY}/tmp_short.log"
  script -c\
    "python3 run.py -c 128 120 -j 12 -w 8 -i 0.00038 --disparityweight 0.24045\
    -s 0.44162 --eps 75.06 --min-size 1 --size-filter 57 --evaluate --clean\
     ${DATA_DIRECTORY}/short_test/left_org/" ${LOG_FILE}\
     | surpress_output ${VERBOSE}
  TEST_VALUES="$(awk '/^Semantic score/ {print 1.0*$4 " " 1.0*$8 " " 1.0*$12}' ${LOG_FILE})"
  TRUE_VALUES="0.622 0.139 1271.2"
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

