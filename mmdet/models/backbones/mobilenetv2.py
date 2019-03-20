import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (xavier_init, constant_init, kaiming_init,
                      normal_init)
from mmcv.runner import load_checkpoint
from ..registry import BACKBONES

def conv_bn(inp, oup, stride, groups=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup, groups=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, ssd_output=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = [
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        self.ssd_output = ssd_output
        if self.ssd_output:
            self.conv1 = nn.Sequential(*self.conv[0 : 3])
            self.conv2 = nn.Sequential(*self.conv[3 :])
        else:
            self.conv = nn.Sequential(*self.conv)

    def compute_conv(self, x):
        if not self.ssd_output:
            return self.conv(x)

        expanded_x = self.conv1(x)
        return (expanded_x, self.conv2(expanded_x))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.compute_conv(x)
        else:
            return self.compute_conv(x)

@BACKBONES.register_module
class SSDMobilenetV2(nn.Module):
    def __init__(self, input_size):
        super(SSDMobilenetV2, self).__init__()
        self.input_size = input_size

        width_mult = 1.0
        block = InvertedResidual
        input_channel = 32
        last_channel = 320
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.bn_first = nn.BatchNorm2d(3)
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                ssd_output = False
                if i == 0 and c == 160 and s == 2:
                    ssd_output = True

                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, t, ssd_output))
                else:
                    self.features.append(block(input_channel, output_channel, 1, t, ssd_output))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.extra_convs = []
        self.extra_convs.append(conv_1x1_bn(last_channel, 1280))#head
        self.extra_convs.append(conv_1x1_bn(1280, 256))
        self.extra_convs.append(conv_bn(256, 256, 2, groups=256))
        self.extra_convs.append(conv_1x1_bn(256, 512, groups=1))#head
        self.extra_convs.append(conv_1x1_bn(512, 128))
        self.extra_convs.append(conv_bn(128, 128, 2, groups=128))
        self.extra_convs.append(conv_1x1_bn(128, 256))#head
        self.extra_convs.append(conv_1x1_bn(256, 128))
        self.extra_convs.append(conv_bn(128, 128, 2, groups=128))
        self.extra_convs.append(conv_1x1_bn(128, 256))#head
        self.extra_convs.append(conv_1x1_bn(256, 64))
        self.extra_convs.append(conv_bn(64, 64, 2, groups=64))
        self.extra_convs.append(conv_1x1_bn(64, 128))#head
        self.extra_convs = nn.Sequential(*self.extra_convs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            state_dict = torch.load(pretrained)
            self.load_state_dict(state_dict, strict=False)
            #load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        x = self.bn_first(x)
        for i, block in enumerate(self.features):
            x = block(x)
            if isinstance(x, tuple):
                outs.append(x[0])
                x = x[1]

        for i, conv in enumerate(self.extra_convs):
            x = conv(x)
            if i % 3 == 0:
                outs.append(x)

        return tuple(outs)
