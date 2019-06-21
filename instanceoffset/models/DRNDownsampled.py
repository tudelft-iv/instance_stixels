# This file is part of Instance Stixels:
# https://github.com/tudelft-iv/instance-stixels
#
# Original:
# Copyright (c) 2017, Fisher Yu
# BSD 3-Clause License
# https://github.com/fyu/drn/blob/16acdba72f4115992e02a22be7e08cb3762f8e51/segment.py#L81
#
# Modifications:
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

import math

import torch
from torch import nn
from torch.nn import functional as F

from . import drn

class DRNDownsampledCombined(nn.Module):
    def __init__(self, model_name, classes, seg_dict, reg_dict):
        super(DRNDownsampledCombined, self).__init__()
        self.seg = DRNSegDownsampled(
                        model_name='drn_d_22',
                        classes=classes,
                        pretrained_model=None)
        self.seg.load_state_dict(seg_dict)
        self.reg = DRNRegressionDownsampled(
                        model_name='drn_d_22',
                        classes=classes)
        self.reg.load_state_dict(reg_dict)

    def forward(self, x):
        s = self.seg(x)
        r = self.reg(x)
        return torch.cat((s, r), dim=1)

class DRNSegDownsampled(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 use_torch_up=False):
        super(DRNSegDownsampled, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=False, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()

        self.up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                     output_padding=0, groups=classes,
                                     bias=False)

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        return self.logsoftmax(x) 

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


class DRNRegressionDownsampled(nn.Module):
    def __init__(self, model_name, classes, pretrained_dict=None,
                 pretrained=True, use_torch_up=False):
        super(DRNRegressionDownsampled, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=False, num_classes=1000)
        self.base = nn.Sequential(*list(model.children())[:-2])

        if pretrained_dict is not None:
            # Hacky way to load pretrained weights and modify layers afterwards.
            self.seg = nn.Conv2d(model.out_dim, classes,
                                 kernel_size=1, bias=True)
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            self.up = up

            self.load_state_dict(pretrained_dict)

            del self.seg
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            self.up = up


        #self.logsoftmax = nn.LogSoftmax(dim=1)
        self.regression = nn.Conv2d(model.out_dim, 2,
                                    kernel_size=1, bias=True)

    def forward(self, x):
        x = self.base(x)
        #x = self.seg(x)
        y = self.regression(x)
        #y = self.up(x)
        return y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


