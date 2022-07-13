from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

from .spherical_conv import *

def sphereConvbn(in_height, in_width, sphereType, in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
            SphereConv(in_height,
                                 in_width,
                                 sphereType,
                                 in_planes,
                                 out_planes,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=dilation if dilation > 1 else pad,
                                 dilation=dilation,
                                 bias=False),
            nn.BatchNorm2d(out_planes))

class RegularBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(RegularBasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out

class SphereBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_height, in_width, sphereType, inplanes, planes, stride, downsample, pad, dilation):
        super(SphereBasicBlock, self).__init__()

        self.conv1 = nn.Sequential(sphereConvbn(in_height, in_width, sphereType, inplanes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True))

        self.conv2 = sphereConvbn(in_height // stride, in_width // stride, sphereType, planes, planes, 3, 1, pad, dilation)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out

class RegularBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(RegularBottleneck, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 1, 1, 0, 1), nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn(planes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True))

        self.conv3 = convbn(planes, planes * self.expansion, 1, 1, 0, 1)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out

class SphereBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_height, in_width, sphereType, inplanes, planes, stride, downsample, pad, dilation):
        super(SphereBottleneck, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 1, 1, 0, 1), nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(sphereConvbn(in_height, in_width, sphereType, planes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True))

        self.conv3 = convbn(planes, planes * self.expansion, 1, 1, 0, 1)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=dilation if dilation > 1 else pad,
                  dilation=dilation,
                  bias=False), nn.BatchNorm2d(out_planes))

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(
        nn.Conv3d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  padding=pad,
                  stride=stride,
                  bias=False), nn.BatchNorm3d(out_planes))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes, planes, 3, stride, pad, dilation),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            convbn(planes, planes, 3, 1, pad, dilation),
            nn.ReLU(inplace=True))

        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes, planes, 3, stride, pad, dilation),
            nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

# MODE-Net-stage1

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    def forward(self, x):
        out = torch.sum(x*self.disp.data,1, keepdim=True)
        return out

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(ResBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(ResBlock, 64, 16, 2,1,1) 
        self.layer3 = self._make_layer(ResBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(ResBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners=True)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature

class sphere_feature_extraction(nn.Module):
    def __init__(self, in_height, in_width, sphereType):
        super(sphere_feature_extraction, self).__init__()
        # print("using feature extraction 4")
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 7, 2, 3, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(RegularBasicBlock, in_height // 2, in_width // 2, sphereType, 32, 64, 3, 1, 1, 1)  #16x4
        self.layer2 = self._make_layer(RegularBasicBlock, in_height // 2, in_width // 2, sphereType, 64, 64, 8, 2, 1, 1)  #32x4
        self.layer3 = self._make_layer(RegularBasicBlock, in_height // 4, in_width // 4, sphereType, 64, 64, 4, 1, 1, 2)  # regular dilation

        self.layer4 = self._make_layer(SphereBasicBlock, in_height // 4, in_width // 4, sphereType, 64, 128, 8, 1, 1, 1)  # sphere
        #self.layer5 = self._make_layer(RegularBottleneck, in_height // 4, in_width // 4, sphereType, 64, 32, 8, 1, 1, 1)

        # self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))  # 1x1 sphere has same behavior with regular conv

        # self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))

        # self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))

        # self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(256, 128, 1, 1, 0, 1), nn.ReLU(inplace=True), convbn(128, 128, 3, 1, 1, 1), nn.ReLU(inplace=True), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))

    def _make_layer(self, block, height, width, sphereType, inplanes, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(inplanes,
                                        planes * block.expansion,
                                        kernel_size=1,
                                        stride=stride,
                                        bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if block == SphereBasicBlock or block == SphereBottleneck:
            print("add sphere block. num: {}, inplanes: {}, planes: {}".format(blocks, inplanes, planes))
            layers.append(block(height, width, sphereType, inplanes, planes, stride, downsample, pad, dilation))
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(height // stride, width // stride, sphereType, inplanes, planes, 1, None, pad, dilation))
        else:
            print("add regular block. num: {}, inplanes: {}, planes: {}".format(blocks, inplanes, planes))
            layers.append(block(inplanes, planes, stride, downsample, pad, dilation))
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output_regualr = self.layer3(output_raw)
        output_sphere = self.layer4(output_regualr)

        # output_branch1 = self.branch1(output_skip)
        # output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

        # output_branch2 = self.branch2(output_skip)
        # output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

        # output_branch3 = self.branch3(output_skip)
        # output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

        # output_branch4 = self.branch4(output_skip)
        # output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

        # output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = torch.cat((output_raw, output_regualr, output_sphere), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


# MODE-Net-stage2

class feature_extraction_Baseline(nn.Module):
    def __init__(self, maxdepth):
        super(feature_extraction_Baseline, self).__init__()
        self.inplanes = 6

        self.layer1 = self._make_layer(BasicBlock, 32, 2, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, 1,1,1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 256, 1, 1,1,1)
        self.layer5 = self._make_layer(BasicBlock, 128, 1, 1,1,1)
        self.layer6 = self._make_layer(BasicBlock, 64, 1, 1,1,1)
        self.layer7 = self._make_last_layer(BasicBlock, 32, 2, 1,1,1)

        self.maxdepth = torch.tensor(maxdepth)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_last_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))
        
        layers.append(nn.Conv2d(planes, 1, kernel_size=1, padding=0, stride = 1, bias=True))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return out * self.maxdepth

class feature_extraction_Unet(nn.Module):
    def __init__(self, maxdepth, channels):
        super(feature_extraction_Unet, self).__init__()
        self.inplanes = 6

        self.layer1 = self._make_layer(BasicBlock, channels[0], 2, 1,1,1)
        self.layer2 = self._make_layer_down(BasicBlock, channels[1], 1, 1,1,1)
        self.layer3 = self._make_layer_down(BasicBlock, channels[2], 1, 1,1,1)

        self.layer4 = self._make_layer_down_up(BasicBlock, channels[3], 1, 1,1,1)
        self.layer5 = self._make_layer_up(BasicBlock, channels[2], 1, 1,1,1)
        self.layer6 = self._make_layer_up(BasicBlock, channels[1], 1, 1,1,1)
        self.layer7 = self._make_last_layer(BasicBlock, channels[0], 2, 1,1,1)

        # self.layer7 = nn.Sequential(
        #     convbn(self.inplanes, channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     convbn(channels[0], channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     convbn(channels[0], channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     convbn(channels[0], channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channels[0], 1, kernel_size=1, padding=0, stride = 1, bias=True),
        #     nn.Sigmoid())
        
        self.maxdepth = torch.tensor(maxdepth)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_layer_down(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(nn.MaxPool2d(2, stride=2))
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_layer_down_up(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(nn.MaxPool2d(2, stride=2))
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))
        layers.append(nn.ConvTranspose2d(
            planes, int(planes/2), 2, 2
        ))
        layers.append(nn.BatchNorm2d(int(planes/2)))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_layer_up(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))
        layers.append(nn.ConvTranspose2d(
            planes, int(planes/2), 2, 2
        ))
        layers.append(nn.BatchNorm2d(int(planes/2)))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_last_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))
        
        layers.append(nn.Conv2d(planes, 1, kernel_size=1, padding=0, stride = 1, bias=True))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(torch.cat((x3, x4), 1))
        x6 = self.layer6(torch.cat((x2, x5), 1))
        x7 = self.layer7(torch.cat((x1, x6), 1))

        return x7 * self.maxdepth

class feature_extraction_UnetRgb(nn.Module):
    def __init__(self, maxdepth, channels):
        super(feature_extraction_UnetRgb, self).__init__()
        self.depth_inplanes = 6
        self.rgb_inplanes = 12

        self.depth_layer1 = self._make_depth_layer(BasicBlock, channels[0], 2, 1,1,1)
        self.depth_layer2 = self._make_depth_layer_down(BasicBlock, channels[1], 1, 1,1,1)
        self.depth_layer3 = self._make_depth_layer_down(BasicBlock, channels[2], 1, 1,1,1)

        self.rgb_layer1 = self._make_rgb_layer(BasicBlock, channels[0], 2, 1,1,1)
        self.rgb_layer2 = self._make_rgb_layer_down(BasicBlock, channels[1], 1, 1,1,1)
        self.rgb_layer3 = self._make_rgb_layer_down(BasicBlock, channels[2], 1, 1,1,1)

        self.fusion_layer1 = self._make_fusion_layer(BasicBlock, channels[0], 2, 1,1,1)
        self.fusion_layer2 = self._make_fusion_layer(BasicBlock, channels[1], 2, 1,1,1)
        self.fusion_layer3 = self._make_fusion_layer(BasicBlock, channels[2], 2, 1,1,1)

        self.depth_layer4 = self._make_depth_layer_down_up(BasicBlock, channels[3], 1, 1,1,1)
        self.depth_layer5 = self._make_depth_layer_up(BasicBlock, channels[2], 1, 1,1,1)
        self.depth_layer6 = self._make_depth_layer_up(BasicBlock, channels[1], 1, 1,1,1)

        self.depth_layer7 = self._make_last_layer(BasicBlock, channels[0], 2, 1,1,1)

        # self.depth_layer7 = nn.Sequential(
        #     convbn(self.depth_inplanes, channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     convbn(channels[0], channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     convbn(channels[0], channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     convbn(channels[0], channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channels[0], 1, kernel_size=1, padding=0, stride = 1, bias=True),
        #     nn.Sigmoid())
        
        self.maxdepth = torch.tensor(maxdepth)
    
    def _make_depth_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.depth_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.depth_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
        self.depth_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.depth_inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_rgb_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.rgb_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.rgb_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.rgb_inplanes, planes, stride, downsample, pad, dilation))
        self.rgb_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.rgb_inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_depth_layer_down(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.depth_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.depth_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(nn.MaxPool2d(2, stride=2))
        layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
        self.depth_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.depth_inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_rgb_layer_down(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.rgb_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.rgb_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(nn.MaxPool2d(2, stride=2))
        layers.append(block(self.rgb_inplanes, planes, stride, downsample, pad, dilation))
        self.rgb_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.rgb_inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_fusion_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = nn.Sequential(
            nn.Conv2d(int(2*planes), planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(int(2*planes), planes, stride, downsample, pad, dilation))
        for i in range(1, blocks):
            layers.append(block(planes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_depth_layer_down_up(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.depth_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.depth_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(nn.MaxPool2d(2, stride=2))
        layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
        self.depth_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.depth_inplanes, planes,1,None,pad,dilation))
        layers.append(nn.ConvTranspose2d(
            planes, int(planes/2), 2, 2
        ))
        layers.append(nn.BatchNorm2d(int(planes/2)))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_depth_layer_up(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.depth_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.depth_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
        self.depth_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.depth_inplanes, planes,1,None,pad,dilation))
        layers.append(nn.ConvTranspose2d(
            planes, int(planes/2), 2, 2
        ))
        layers.append(nn.BatchNorm2d(int(planes/2)))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_last_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.depth_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.depth_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
        self.depth_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.depth_inplanes, planes,1,None,pad,dilation))
        
        layers.append(nn.Conv2d(planes, 1, kernel_size=1, padding=0, stride = 1, bias=True))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, depth_input, rgb_input):
        # depth encoder
        depth1 = self.depth_layer1(depth_input)
        depth2 = self.depth_layer2(depth1)
        depth3 = self.depth_layer3(depth2)
        depth4 = self.depth_layer4(depth3)
        # rgb encoder
        rgb1 = self.rgb_layer1(rgb_input)
        rgb2 = self.rgb_layer2(rgb1)
        rgb3 = self.rgb_layer3(rgb2)
        # feature fusion of depth & rgb
        fusion1 = self.fusion_layer1(torch.cat((depth1, rgb1), 1))
        fusion2 = self.fusion_layer2(torch.cat((depth2, rgb2), 1))
        fusion3 = self.fusion_layer3(torch.cat((depth3, rgb3), 1))
        # decoder
        depth5 = self.depth_layer5(torch.cat((fusion3, depth4), 1))
        depth6 = self.depth_layer6(torch.cat((fusion2, depth5), 1))
        depth7 = self.depth_layer7(torch.cat((fusion1, depth6), 1))

        return depth7 * self.maxdepth

class feature_extraction_UnetRgbConf(nn.Module):
    def __init__(self, maxdepth, channels, inplanes):
        super(feature_extraction_UnetRgbConf, self).__init__()
        self.depth_inplanes = inplanes['depth']
        self.rgb_inplanes = inplanes['rgb']

        self.depth_layer1 = self._make_depth_layer(BasicBlock, channels[0], 2, 1,1,1)
        self.depth_layer2 = self._make_depth_layer_down(BasicBlock, channels[1], 1, 1,1,1)
        self.depth_layer3 = self._make_depth_layer_down(BasicBlock, channels[2], 1, 1,1,1)

        self.rgb_layer1 = self._make_rgb_layer(BasicBlock, channels[0], 2, 1,1,1)
        self.rgb_layer2 = self._make_rgb_layer_down(BasicBlock, channels[1], 1, 1,1,1)
        self.rgb_layer3 = self._make_rgb_layer_down(BasicBlock, channels[2], 1, 1,1,1)

        self.fusion_layer1 = self._make_fusion_layer(BasicBlock, channels[0], 2, 1,1,1)
        self.fusion_layer2 = self._make_fusion_layer(BasicBlock, channels[1], 2, 1,1,1)
        self.fusion_layer3 = self._make_fusion_layer(BasicBlock, channels[2], 2, 1,1,1)

        self.depth_layer4 = self._make_depth_layer_down_up(BasicBlock, channels[3], 1, 1,1,1)
        self.depth_layer5 = self._make_depth_layer_up(BasicBlock, channels[2], 1, 1,1,1)
        self.depth_layer6 = self._make_depth_layer_up(BasicBlock, channels[1], 1, 1,1,1)

        self.depth_layer7 = self._make_last_layer(BasicBlock, channels[0], 2, 1,1,1)

        # self.depth_layer7 = nn.Sequential(
        #     convbn(self.depth_inplanes, channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     convbn(channels[0], channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     convbn(channels[0], channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     convbn(channels[0], channels[0], 3, 1, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channels[0], 1, kernel_size=1, padding=0, stride = 1, bias=True),
        #     nn.Sigmoid())
        
        self.maxdepth = torch.tensor(maxdepth)
    
    def _make_depth_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.depth_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.depth_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
        self.depth_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.depth_inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_rgb_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.rgb_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.rgb_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.rgb_inplanes, planes, stride, downsample, pad, dilation))
        self.rgb_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.rgb_inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_depth_layer_down(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.depth_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.depth_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(nn.MaxPool2d(2, stride=2))
        layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
        self.depth_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.depth_inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_rgb_layer_down(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.rgb_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.rgb_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(nn.MaxPool2d(2, stride=2))
        layers.append(block(self.rgb_inplanes, planes, stride, downsample, pad, dilation))
        self.rgb_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.rgb_inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_fusion_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = nn.Sequential(
            nn.Conv2d(int(2*planes), planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(int(2*planes), planes, stride, downsample, pad, dilation))
        for i in range(1, blocks):
            layers.append(block(planes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def _make_depth_layer_down_up(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.depth_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.depth_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(nn.MaxPool2d(2, stride=2))
        layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
        self.depth_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.depth_inplanes, planes,1,None,pad,dilation))
        layers.append(nn.ConvTranspose2d(
            planes, int(planes/2), 2, 2
        ))
        layers.append(nn.BatchNorm2d(int(planes/2)))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_depth_layer_up(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.depth_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.depth_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
        self.depth_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.depth_inplanes, planes,1,None,pad,dilation))
        layers.append(nn.ConvTranspose2d(
            planes, int(planes/2), 2, 2
        ))
        layers.append(nn.BatchNorm2d(int(planes/2)))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_last_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.depth_inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.depth_inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
        self.depth_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.depth_inplanes, planes,1,None,pad,dilation))
        
        layers.append(nn.Conv2d(planes, 1, kernel_size=1, padding=0, stride = 1, bias=True))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, depth_input, rgb_input):
        # depth encoder
        depth1 = self.depth_layer1(depth_input)
        depth2 = self.depth_layer2(depth1)
        depth3 = self.depth_layer3(depth2)
        depth4 = self.depth_layer4(depth3)
        # rgb encoder
        rgb1 = self.rgb_layer1(rgb_input)
        rgb2 = self.rgb_layer2(rgb1)
        rgb3 = self.rgb_layer3(rgb2)
        # feature fusion of depth & rgb
        fusion1 = self.fusion_layer1(torch.cat((depth1, rgb1), 1))
        fusion2 = self.fusion_layer2(torch.cat((depth2, rgb2), 1))
        fusion3 = self.fusion_layer3(torch.cat((depth3, rgb3), 1))
        # decoder
        depth5 = self.depth_layer5(torch.cat((fusion3, depth4), 1))
        depth6 = self.depth_layer6(torch.cat((fusion2, depth5), 1))
        depth7 = self.depth_layer7(torch.cat((fusion1, depth6), 1))

        return depth7 * self.maxdepth


class depth_regression(nn.Module):
    def __init__(self, maxdepth):
        super(depth_regression, self).__init__()
        self.lastconv = nn.Sequential(convbn(64, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(32, 1, kernel_size=1, padding=0, stride = 1, bias=True),
                                        nn.Sigmoid())
        self.maxdepth = torch.tensor(maxdepth)

    def forward(self, x):
        out = self.lastconv(x)
        return out * self.maxdepth