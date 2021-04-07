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

#import ipdb

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
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
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        #return self.logsoftmax(y), x
        return self.logsoftmax(y), self.up(x)

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

class DRNDoubleSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_dict=None,
                 pretrained=True, use_torch_up=False):
        super(DRNDoubleSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
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

            # Extend segmentation layer by 2 channels.
            old_seg = self.seg
            self.seg = nn.Conv2d(model.out_dim, classes+2,
                                 kernel_size=1, bias=True)
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
            m.weight.data[:classes] = old_seg.weight.data
            m.bias.data[:classes] = old_seg.bias.data
        else:
            # Extend segmentation layer by 2 channels.
            self.seg = nn.Conv2d(model.out_dim, classes+2,
                                 kernel_size=1, bias=True)
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()

        self.logsoftmax = nn.LogSoftmax(dim=1)

        # Extend upsampling layer.
        up = nn.ConvTranspose2d(classes+2, classes+2, 16, stride=8, padding=4,
                                output_padding=0, groups=classes+2,
                                bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        #return self.logsoftmax(y), x
        y = torch.cat((self.logsoftmax(y[:,:-2]), y[:,-2:]), dim=1)
        #return self.logsoftmax(y)#, self.up(x)
        return y

    def optim_seg_parameters(self, memo=None):
        for param in self.base.parameters():
            param.requires_grad = False
        for param in self.seg.parameters():
            yield param

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

class DRNOffsetDisparity(nn.Module):
    def __init__(self, model_name, classes, pretrained_dict=None,
                 pretrained=True, use_torch_up=False):
        super(DRNOffsetDisparity, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
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

            # Extend segmentation layer by 2 channels.
            old_seg = self.seg
            self.seg = nn.Conv2d(model.out_dim, classes+3,
                                 kernel_size=1, bias=True)
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
            m.weight.data[:classes] = old_seg.weight.data
            m.bias.data[:classes] = old_seg.bias.data
        else:
            # Extend segmentation layer by 2 channels.
            self.seg = nn.Conv2d(model.out_dim, classes+3,
                                 kernel_size=1, bias=True)
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()

        self.logsoftmax = nn.LogSoftmax(dim=1)

        # Extend upsampling layer.
        up = nn.ConvTranspose2d(classes+3, classes+3, 16, stride=8, padding=4,
                                output_padding=0, groups=classes+3,
                                bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        #return self.logsoftmax(y), x
        if not self.training:
            y[:,-3] = torch.clamp(y[:,-3], 0, 128)
        y = torch.cat((self.logsoftmax(y[:,:-3]), y[:,-3:]), dim=1)
        #return self.logsoftmax(y)#, self.up(x)
        return y

    def optim_seg_parameters(self, memo=None):
        for param in self.base.parameters():
            param.requires_grad = False
        for param in self.seg.parameters():
            yield param

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

class DRNRegressionOnly(nn.Module):
    def __init__(self, model_name, classes, pretrained_dict=None,
                 pretrained=True, use_torch_up=False):
        super(DRNRegressionOnly, self).__init__()
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

            # Extend segmentation layer by 2 channels.
            old_seg = self.seg
            self.seg = nn.Conv2d(model.out_dim, classes+2,
                                 kernel_size=1, bias=True)
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
            m.weight.data[:classes] = old_seg.weight.data
            m.bias.data[:classes] = old_seg.bias.data
        else:
            # Extend segmentation layer by 2 channels.
            self.seg = nn.Conv2d(model.out_dim, classes+2,
                                 kernel_size=1, bias=True)
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()

        del self.seg
        #self.logsoftmax = nn.LogSoftmax(dim=1)
        self.regression = nn.Conv2d(model.out_dim, 2,
                                    kernel_size=1, bias=True)

        # Extend upsampling layer.
        up = nn.ConvTranspose2d(2, 2, 16, stride=8, padding=4,
                                output_padding=0, #groups=2,
                                bias=False)
        fill_up_weights(up)
        #up.weight.requires_grad = False
        self.up = up

    def forward(self, x):
        x = self.base(x)
        #x = self.seg(x)
        x = self.regression(x)
        y = self.up(x)
        return y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

class DRNMultifield(nn.Module):
    def __init__(self, model_name, classes, fields=4, pretrained_dict=None,
                 pretrained=True, use_torch_up=False):
        super(DRNMultifield, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=False, num_classes=1000)
        self.base = nn.Sequential(*list(model.children())[:-2])
        self.fields = fields

        if pretrained_dict is not None:
            # Load pretrained weights and modify layers afterwards.
            self.seg = nn.Conv2d(model.out_dim, classes,
                                 kernel_size=1, bias=True)
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            self.up = up

            self.load_state_dict(pretrained_dict)
        else:
            # Extend segmentation layer by 2 channels.
            self.seg = nn.Conv2d(model.out_dim,
                                 classes,
                                 kernel_size=1, bias=True)
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()

        self.hidden = nn.Conv2d(model.out_dim,
                                model.out_dim,
                                kernel_size=1, bias=True)
        self.x_field = nn.Conv2d(model.out_dim,
                                 fields,
                                 kernel_size=1, bias=True)
        self.y_field = nn.Conv2d(model.out_dim,
                                 fields,
                                 kernel_size=1, bias=True)
        self.field_activation = nn.Conv2d(model.out_dim,
                                          fields,
                                          kernel_size=1, bias=True)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax2d()
        self.steepness = None
        if True:
            self.steepness = nn.Parameter(torch.ones(1))

        # Setup upsampling layer.
        up = nn.ConvTranspose2d(classes+2,
                                classes+2,
                                16, stride=8, padding=4,
                                output_padding=0,
                                groups=classes+2,
                                bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        self.up = up

        # Setup upsampling layer.
        up = nn.ConvTranspose2d(fields,
                                fields,
                                16, stride=8, padding=4,
                                output_padding=0,
                                groups=fields,
                                bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        self.fields_up = up

    def forward(self, x, all_fields=False):
        x = self.base(x)
        h = F.relu(self.hidden(x))
        x_fields = self.x_field(h)
        y_fields = self.y_field(h)
        x_field = x_fields
        y_field = y_fields
        if not all_fields:
            del x_fields, y_fields

        field_activation = self.softmax(self.field_activation(h))
        if self.steepness is not None:
            field_activation = self.steepness * field_activation
        x_field = field_activation * x_field
        x_field = torch.sum(x_field, dim=1, keepdim=True)
        y_field = field_activation * y_field
        y_field = torch.sum(y_field, dim=1, keepdim=True)
        x = self.seg(x)
        x = torch.cat((x,y_field,x_field), dim=1)

        y = self.up(x)
        #return self.logsoftmax(y), x
        y = torch.cat((self.logsoftmax(y[:,:-2]), y[:,-2:]), dim=1)
        #return self.logsoftmax(y)#, self.up(x)
        if all_fields:
            x_fields = self.fields_up(x_fields)
            y_fields = self.fields_up(y_fields)
            field_activation = self.fields_up(field_activation)
            return y, y_fields, x_fields, field_activation
        return y

    def optim_field_parameters(self):
        for param in self.base.parameters():
            param.requires_grad = False
        for param in self.seg.parameters():
            param.requires_grad = False

        for param in self.x_field.parameters():
            yield param
        for param in self.y_field.parameters():
            yield param
        for param in self.field_activation.parameters():
            yield param
        if self.steepness is not None:
            yield self.steepness

    def optim_seg_parameters(self, memo=None):
        for param in self.base.parameters():
            param.requires_grad = False

        for param in self.seg.parameters():
            yield param

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param
        for param in self.x_field.parameters():
            yield param
        for param in self.y_field.parameters():
            yield param
        for param in self.field_activation.parameters():
            yield param
        if self.steepness is not None:
            yield self.steepness

class DRNMultifieldAfterUp(nn.Module):
    def __init__(self, model_name, classes, fields=4, pretrained_dict=None,
                 pretrained=True, use_torch_up=False):
        super(DRNMultifieldAfterUp, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=False, num_classes=1000)
        self.base = nn.Sequential(*list(model.children())[:-2])
        self.fields = fields

        if pretrained_dict is not None:
            # Load pretrained weights and modify layers afterwards.
            self.seg = nn.Conv2d(model.out_dim, classes,
                                 kernel_size=1, bias=True)
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            self.up = up

            self.load_state_dict(pretrained_dict)
        else:
            self.seg = nn.Conv2d(model.out_dim,
                                 classes,
                                 kernel_size=1, bias=True)
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()

        self.x_field = nn.Conv2d(model.out_dim,
                                 fields,
                                 kernel_size=1, bias=True)
        self.y_field = nn.Conv2d(model.out_dim,
                                 fields,
                                 kernel_size=1, bias=True)
        self.field_activation = nn.Conv2d(model.out_dim,
                                          fields,
                                          kernel_size=1, bias=True)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax2d()
        self.steepness = None
        if True:
            self.steepness = nn.Parameter(torch.ones(1))

        # Setup upsampling layer.
        up = nn.ConvTranspose2d(classes,
                                classes,
                                16, stride=8, padding=4,
                                output_padding=0,
                                groups=classes,
                                bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        self.up = up

        # Setup upsampling layer.
        up = nn.ConvTranspose2d(fields,
                                fields,
                                16, stride=8, padding=4,
                                output_padding=0,
                                groups=fields,
                                bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        self.fields_up = up

        if use_torch_up:
            self.up = nn.Upsample(scale_factor=8, mode='bilinear')
            self.fields_up = self.up

    def forward(self, x, all_fields=False):
        x = self.base(x)
        x_fields = self.fields_up(self.x_field(x))
        y_fields = self.fields_up(self.y_field(x))
        x_field = x_fields
        y_field = y_fields
        if not all_fields:
            del x_fields, y_fields

        field_activation = self.softmax(self.field_activation(x))
        field_activation = self.up(field_activation)
        if self.steepness is not None:
            field_activation = self.steepness * field_activation
        x_field = field_activation * x_field
        x_field = torch.sum(x_field, dim=1, keepdim=True)
        y_field = field_activation * y_field
        y_field = torch.sum(y_field, dim=1, keepdim=True)
        x = self.seg(x)
        x = self.up(x)
        y = torch.cat((self.logsoftmax(x),y_field,x_field), dim=1)

        if all_fields:
            return y, y_fields, x_fields, field_activation
        return y

    def optim_field_parameters(self):
        for param in self.base.parameters():
            param.requires_grad = False
        for param in self.seg.parameters():
            param.requires_grad = False

        for param in self.x_field.parameters():
            yield param
        for param in self.y_field.parameters():
            yield param
        for param in self.field_activation.parameters():
            yield param
        if self.steepness is not None:
            yield self.steepness

    def optim_seg_parameters(self, memo=None):
        for param in self.base.parameters():
            param.requires_grad = False

        for param in self.seg.parameters():
            yield param

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param
        for param in self.x_field.parameters():
            yield param
        for param in self.y_field.parameters():
            yield param
        for param in self.field_activation.parameters():
            yield param
        if self.steepness is not None:
            yield self.steepness


class DRNMultifieldMax(nn.Module):
    def __init__(self, model_name, classes, fields=4, pretrained_dict=None,
                 pretrained=True, use_torch_up=False):
        super(DRNMultifieldMax, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=False, num_classes=1000)
        self.base = nn.Sequential(*list(model.children())[:-2])
        self.fields = fields

        if pretrained_dict is not None:
            # Load pretrained weights and modify layers afterwards.
            self.seg = nn.Conv2d(model.out_dim, classes,
                                 kernel_size=1, bias=True)
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            self.up = up

            self.load_state_dict(pretrained_dict)
        else:
            # Extend segmentation layer by 2 channels.
            self.seg = nn.Conv2d(model.out_dim,
                                 classes,
                                 kernel_size=1, bias=True)
            m = self.seg
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()

        self.x_field = nn.Conv2d(model.out_dim,
                                 fields,
                                 kernel_size=1, bias=True)
        self.y_field = nn.Conv2d(model.out_dim,
                                 fields,
                                 kernel_size=1, bias=True)
        self.field_activation = nn.Conv2d(model.out_dim,
                                          fields,
                                          kernel_size=1, bias=True)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax2d()

        # Setup upsampling layer.
        up = nn.ConvTranspose2d(classes,
                                classes,
                                16, stride=8, padding=4,
                                output_padding=0,
                                groups=classes,
                                bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        self.up = up

        # Setup upsampling layer.
        up = nn.ConvTranspose2d(fields,
                                fields,
                                16, stride=8, padding=4,
                                output_padding=0,
                                groups=fields,
                                bias=False)
        fill_up_weights(up)
        up.weight.requires_grad = False
        self.fields_up = up

    def forward(self, x, all_fields=False):
        # TODO: double check this function
        x = self.base(x)
        x_fields = self.x_field(x)
        x_fields = self.fields_up(x_fields)
        x_field = x_fields

        y_fields = self.y_field(x)
        y_fields = self.fields_up(y_fields)
        y_field = y_fields

        field_activations = self.softmax(self.field_activation(x))
        field_activations = self.fields_up(field_activations)
        field_activation = field_activations
        field_activation = field_activation.argmax(dim=1, keepdim=True)

        if not all_fields: # only keep full fields if neccessary
            del x_fields, y_fields, field_activations

        x_field = torch.gather(x_field, dim=1, index=field_activation)
        y_field = torch.gather(y_field, dim=1, index=field_activation)

        x = self.seg(x)
        x = self.up(x)

        y = torch.cat((self.logsoftmax(x),y_field,x_field), dim=1)

        if all_fields:
            return y, y_fields, x_fields, field_activations
        return y

    def optim_seg_parameters(self, memo=None):
        for param in self.base.parameters():
            param.requires_grad = False
        for param in self.seg.parameters():
            yield param

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param
        for param in self.x_field.parameters():
            yield param
        for param in self.y_field.parameters():
            yield param
        for param in self.field_activation.parameters():
            yield param

class DRNSeg_inference(nn.Module):
    """
    Simple wrapper around DRNSeg to only return softmax output when forward is
    called.
    """
    def __init__(self, **kwargs):
        super(DRNSeg_inference, self).__init__()
        self.base = DRNSeg(**kwargs)

    def load_state_dict(self, dict_, base=True):
        if base:
            self.base.load_state_dict(dict_)
        else:
            super(DRNSeg_inference, self).load_state_dict(dict_)

    def forward(self, x):
        # return negative log softmax
        return self.base(x)[0]

class DRNInstance(nn.Module):
    """
    Module that adds a regression head on top of the output of the semantic
    segmentation network.
    regression_head : nn.module that is used for regression
    """
    def __init__(self, regression_head, train_all=True, **kwargs):
        # TODO: require pretrained weights?
        super(DRNInstance, self).__init__()
        self.drnseg = DRNSeg(**kwargs)

        self.train_all = train_all
        self.regression_head =\
                regression_head(in_channels=19, out_channels=2)

    # TODO: if you leave it like this, you cannot load the full module later on
    # after saving
    def load_state_dict(self, dict_):
        self.drnseg.load_state_dict(dict_)

    def regression_parameters(self):
        return self.regression_head.parameters()

    def forward(self, x):
        # Let's fix the weights for the semantic part.
        if self.train_all:
            softmax, seg_layer = self.drnseg(x)
        else:
            with torch.no_grad():
                # TODO: maybe do not use DRNSeg.seg output, but work on
                # DRNSeg.base output.
                softmax, seg_layer = self.drnseg(x)
        concat = self.regression_head(seg_layer)
        # return negative log softmax
        concat = torch.cat((softmax, concat), dim=1)

        return concat

class RegressionHead_linearconv(nn.Module):
    """
    Regression head that is capable of estimating the 2d offset of each pixel
    to instance mean.
    """

    def __init__(self, in_channels, out_channels):
        super(RegressionHead_linearconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=5, padding=(5 // 2))

    def forward(self, x):
        x = self.conv(x)
        return x

class RegressionHead_3layers(nn.Module):
    """
    Regression head with 3 layers that is capable of estimating the 2d offset
    of each pixel to instance mean.
    """

    def __init__(self, in_channels, out_channels):
        super(RegressionHead_3layers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=5, padding=(5 // 2))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=5, padding=(5 // 2))
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, padding=0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
