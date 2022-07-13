from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
import random

class Stereo12(nn.Module):
    def __init__(self):
        super(Stereo12, self).__init__()

    def forward(self, depth12, depth13, depth14, depth23, depth24, depth34):
        return depth12

class Stereo13(nn.Module):
    def __init__(self):
        super(Stereo13, self).__init__()

    def forward(self, depth12, depth13, depth14, depth23, depth24, depth34):
        return depth13

class Stereo14(nn.Module):
    def __init__(self):
        super(Stereo14, self).__init__()

    def forward(self, depth12, depth13, depth14, depth23, depth24, depth34):
        return depth14

class StereoFusion0(nn.Module):
    def __init__(self):
        super(StereoFusion0, self).__init__()

        self.feature_extraction = feature_extraction_Fusion0()
        self.depth_regression = depth_regression(maxdepth)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, depth12, depth13, depth14, depth23, depth24, depth34):
        input_feature = torch.cat((depth12, depth12, depth12, depth12, depth12, depth12), 1)

        input_feature  = self.feature_extraction(input_feature)

        pred = self.depth_regression(input_feature)

        return pred

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
        # depth 为 0 的点直接赋值为其他视角下该点的非零值的平均（感觉费时不讨好，先不实现）
        input_feature = torch.cat(depthes, 1)

        pred  = self.feature_extraction(input_feature)

        return pred

class MultiviewFusion0(nn.Module):
    def __init__(self, maxdepth):
        super(MultiviewFusion0, self).__init__()

        self.feature_extraction = feature_extraction_Fusion0()
        self.depth_regression = depth_regression(maxdepth)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, depth12, depth13, depth14, depth23, depth24, depth34):
        # depth 为 0 的点直接赋值为其他视角下该点的非零值的平均（感觉费时不讨好，先不实现）
        input_feature = torch.cat((depth12, depth13, depth14, depth23, depth24, depth34), 1)

        input_feature  = self.feature_extraction(input_feature)

        pred = self.depth_regression(input_feature)

        return pred

class MultiviewFusion1(nn.Module):
    def __init__(self, maxdepth):
        super(MultiviewFusion1, self).__init__()

        self.feature_extraction = feature_extraction_Fusion1()
        self.depth_regression = depth_regression(maxdepth)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, depth12, depth13, depth14, depth23, depth24, depth34):
        # depth 为 0 的点直接赋值为其他视角下该点的非零值的平均（感觉费时不讨好，先不实现）
        input_feature = torch.cat((depth12, depth13, depth14, depth23, depth24, depth34), 1)

        input_feature  = self.feature_extraction(input_feature)

        pred = self.depth_regression(input_feature)

        return pred

class MultiviewFusion2(nn.Module):
    def __init__(self, maxdepth):
        super(MultiviewFusion2, self).__init__()

        self.feature_extraction = feature_extraction_Fusion0()
        self.depth_regression = depth_regression(maxdepth)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, depth12, depth13, depth14, depth23, depth24, depth34):
        # depth 为 0 的点直接赋值为其他视角下该点的非零值的平均（感觉费时不讨好，先不实现）
        input_feature = torch.cat((depth12, depth13, depth14, depth23, depth24, depth34), 1)
        index = [i for i in range(6)]
        random.shuffle(index)
        input_feature = input_feature[:,index,:,:]

        input_feature  = self.feature_extraction(input_feature)

        pred = self.depth_regression(input_feature)

        return pred

class Unet(nn.Module):
    def __init__(self, maxdepth, channels=[32, 64, 128, 256]):
        super(Unet, self).__init__()

        self.feature_extraction = feature_extraction_Unet(maxdepth, channels)

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

class UnetRgb(nn.Module):
    def __init__(self, maxdepth, channels):
        super(UnetRgb, self).__init__()

        self.feature_extraction = feature_extraction_UnetRgb(maxdepth, channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, depthes, rgbs):
        depth_input = torch.cat(depthes, 1)
        rgb_input = torch.cat(rgbs, 1)

        pred  = self.feature_extraction(depth_input, rgb_input)

        return pred

class UnetRgbConf(nn.Module):
    def __init__(self, maxdepth, channels, inplanes):
        super(UnetRgbConf, self).__init__()

        self.feature_extraction = feature_extraction_UnetRgbConf(maxdepth, channels, inplanes)

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
