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


import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import math

class ZeroMeanUnitVarModel(nn.Module):
    def __init__(self, model, means, std):
        super(ZeroMeanUnitVarModel, self).__init__()
        self.model = model
        self.means = means
        self.std = std

    def forward(self, x):
        print("zv1 x.shape =",x.shape)
        x = F.normalize(x.squeeze(), self.means, self.std).unsqueeze(0)
        print("zv2 x.shape =",x.shape)
        return self.model(x)

class FlipAndPad(nn.Module):
    def __init__(self, model, output_height=98, classification_dim=19):
        super(FlipAndPad, self).__init__()
        self.model = model
        self.classification_dim = classification_dim
        self.pad_rows = 2**(math.ceil(math.log2(output_height+1)))\
                        - output_height
        print("pad_rows = ",self.pad_rows)

    def forward(self, x):
        print("fp1 x.shape =",x.shape)
        x = self.model(x)
        #~print("fp2 x.shape =",x.shape)
        #~#x[:self.classification_dim] = -x[:self.classification_dim]
        #~print("fp3 x.shape =",x.shape)
        x = x.permute(0,3,1,2)
        # If flip export does not work properly, work around for now using
        # index_select.
        #x = x.flip(3)
        print("fp4 x.shape =",x.shape)
        x = torch.index_select(x, 3, torch.arange(97,-1,-1).cuda())
        x = nn.functional.pad(x, value=0,
                              pad=(0,self.pad_rows))# pad last dim "left/right"
        print("fp5 x.shape =",x.shape)
        x *= 8
        x = x.int()
        return x
