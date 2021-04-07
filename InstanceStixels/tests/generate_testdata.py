#!/usr/bin/env python3

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

import h5py
import numpy as np

# Parameters to be chosen
shape = (5, 9, 3)
stixel_width = 3
data_pattern = "random" # random, channel_id

print("Generating test data and saving in testdata.h")

# Filenames
filename = "testdata"
h5_filename = "{}.h5".format(filename)
header_filename = "{}.h".format(filename)

# Generating data. Different patterns might be useful for debugging.
m = np.random.rand(*shape)
if data_pattern == "channel_id":
    m = np.ones(shape)
    for i in range(m.shape[2]):
        m[:,:,i] *= i+1

# --- data for "segmentation" tests
# write the matrix to the test file
h5f = h5py.File(h5_filename,'w')
h5f['nlogprobs'] = m
h5f.close()

# --- data for "stixelskernels" tests
# compute padding to fit to next higher power of 2
rows_power2 = int(2**np.ceil(np.log2(m.shape[0]+1)))
# join columns using means
m_joined = np.zeros((m.shape[1] // 3, m.shape[2], rows_power2))
for stixel in range(m.shape[1] // stixel_width):
    start = stixel * stixel_width
    end = start + stixel_width
    m_joined[stixel,:,m.shape[0]-1::-1] = m[:,start:end,:].sum(axis=1).T

# computing prefix sums
m_prefixsums = np.zeros_like(m_joined)
for class_ in range(m.shape[2]):
    m_prefixsums[:,:,1:] = np.cumsum(m_joined[:,:,:-1], axis=2)

#print(m_prefixsums)
#print(m_joined.flatten())

# write stuff into header to check if reading from h5 file works
with open(header_filename,'w') as f:
    f.write("#include <array>\n")
    f.write("#include <string>\n")
    f.write("\n")
    f.write("namespace TESTDATA {\n")
    f.write("\n")
    f.write("const std::string h5_filename = \"../tests/{}\";\n".\
          format(h5_filename))
    f.write("const int stixel_width = {};\n".format(stixel_width))
    f.write("\n")
    f.write("const std::array<float, {}> flat = {{ {} }};\n".\
          format(np.prod(shape), repr(m.flatten().tolist())[1:-1])) 
    f.write("\n")
    f.write("const std::array<float, {}> joined = {{ {} }};\n".\
          format(np.prod(m_joined.shape), 
                 repr(m_joined.flatten().tolist())[1:-1]))
    f.write("\n")
    f.write("const std::array<float, {}> prefixsums = {{ {} }};\n".\
          format(np.prod(m_prefixsums.shape), 
                 repr(m_prefixsums.flatten().tolist())[1:-1]))
    f.write("};\n")
