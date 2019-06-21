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

import os
import sys
import argparse
import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cs

if __name__ == "__main__":
    # This is soooo ugly, but it works.
    gt_path = sys.argv[1]
    results_path = sys.argv[2]
    sys.argv = []

    cs.args.groundTruthSearch =\
            os.path.join(os.path.abspath(gt_path), "*_gtFine_instanceIds.png")
    cs.args.predictionPath = os.path.abspath(results_path)
    print(cs.args.groundTruthSearch)
    print(cs.args.predictionPath)
    cs.main()
