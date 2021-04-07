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


# Running the stixel visualization script in the corresponding conda
# environment.
ARGS="$@"

echo -e "\n--- Visualizing results."
cd $(dirname $0)
source activate instance_stixels
python clustering_visualization.py ${ARGS}
source deactivate

