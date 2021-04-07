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


import argparse
import cv2
import ipdb

def main(FILES, min_value, max_value):
    for filename in FILES:
        img = cv2.imread(filename)
        offset = min_value - 0
        scale = 255/(max_value - min_value)
        img = (img-offset) * scale
        img = img.astype(int)
        print(f"max = {img.max()}")
        print(f"min = {img.min()}")
        cv2.imwrite(filename, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description="Normalize images by stretching from min to max.")
    parser.add_argument("--min","-i", default=0, type=int,
                        help="Min value (default=0).")
    parser.add_argument("--max","-a", default=128, type=int,
                        help="Max value (default=128).")
    parser.add_argument("--noreplace","-n", action="store_true",
                        help="Don't overwrite files (rename).")
    parser.add_argument("FILES", nargs='+', type=str,
                        help="Image files.")

    args = parser.parse_args()

    if args.noreplace:
        raise TypeError("Well, apparently you do need it...")
    main(args.FILES, args.min, args.max)
