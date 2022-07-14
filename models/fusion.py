from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
import random

class Baseline(nn.Module):
    def __init__(self, maxdepth):
        super(Baseline, self).__init__()
        self.feature_extraction = feature_extraction_Baseline(maxdepth)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, depthes):
        input_feature = torch.cat(depthes, 1)
        pred  = self.feature_extraction(input_feature)
        return pred

class MODE_Fusion(nn.Module):
    def __init__(self, maxdepth, channels, inplanes):
        super(MODE_Fusion, self).__init__()
        self.feature_extraction = feature_extraction_MODE_Fusion(maxdepth, channels, inplanes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, depthes, confs, rgbs):
        depthes_confs = []
        for i in range(len(depthes)):
            depthes_confs.append(depthes[i])
            depthes_confs.append(confs[i])
        depth_conf_input = torch.cat(depthes_confs, 1)
        rgb_input = torch.cat(rgbs, 1)

        pred  = self.feature_extraction(depth_conf_input, rgb_input)

        return pred
