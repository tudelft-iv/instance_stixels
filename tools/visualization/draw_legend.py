#!/usr/bin/env python

# Copyright (C) 2019 Thomas Hehn. All right reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Partly adapted from:
# https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib

import numpy as np
from matplotlib import pyplot as plt

# adapted from cityscapes scripts -> helpers -> labels.py
label_colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70],
               # 0 = road, 1 = sidewalk, 2 = building
                [102, 102, 156], [190, 153, 153], [153, 153, 153],
               # 3 = wall, 4 = fence, 5 = pole
                [250, 170, 30], [220, 220, 0], [107, 142, 35],
               # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                [152, 251, 152], [70, 130, 180], [220, 20, 60],
               # 9 = terrain, 10 = sky, 11 = person
                [255, 0, 0], [0, 0, 142], [0, 0, 70],
               # 12 = rider, 13 = car, 14 = truck
                [0, 60, 100], [0, 80, 100], [0, 0, 230],
               # 15 = bus, 16 = train, 17 = motocycle
                [119, 11, 32]]
               # 18 = bicycle
label_colors = np.array(label_colors) / 255.
labels = ["road", "sidewalk", "building", "wall", "fence", "pole",
          "traffic light", "traffic sign", "vegetation", "terrain",
          "sky", "person", "rider", "car", "truck", "bus", "train",
          "motocycle", "bicycle"]

#COLOR_LUT = np.zeros((256,1,3), dtype=np.uint8)
#COLOR_LUT[1:20,0,:] = np.array(label_colors).astype(np.uint8)
#COLOR_LUT = cv2.cvtColor(COLOR_LUT, cv2.COLOR_RGB2BGR)

f = lambda m,c: plt.plot([],[], markersize=12, marker=m, color=c, ls="none")[0]
handles = [f("s", c) for c in label_colors]
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)
plt.axis('off')

def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)
plt.show()


