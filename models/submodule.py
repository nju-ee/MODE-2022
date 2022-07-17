from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

import sys
sys.path.append('../')
from .basic import SphereConv


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

  return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

  return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False), nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True))

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


class disparityregression(nn.Module):
  def __init__(self, maxdisp):
    super(disparityregression, self).__init__()
    self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()

  def forward(self, x):
    out = torch.sum(x * self.disp.data, 1, keepdim=True)
    return out


# NOTE: build feature extraction with sphere conv
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


def sphereConvbnrelu(in_height, in_width, sphereType, in_planes, out_planes, kernel_size, stride, pad, dilation):

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
      nn.BatchNorm2d(out_planes),
      nn.ReLU(inplace=True))


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


# NOTE: currently used feature extraction block
class sphere_feature_extraction(nn.Module):
  def __init__(self, in_height, in_width, sphereType):
    super(sphere_feature_extraction, self).__init__()
    self.inplanes = 32
    self.firstconv = nn.Sequential(convbn(3, 32, 7, 2, 3, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))

    self.layer1 = self._make_layer(RegularBasicBlock, in_height // 2, in_width // 2, sphereType, 32, 64, 3, 1, 1, 1)  #16x4
    self.layer2 = self._make_layer(RegularBasicBlock, in_height // 2, in_width // 2, sphereType, 64, 64, 8, 2, 1, 1)  #32x4
    self.layer3 = self._make_layer(RegularBasicBlock, in_height // 4, in_width // 4, sphereType, 64, 64, 4, 1, 1, 2)  # regular dilation

    self.layer4 = self._make_layer(SphereBasicBlock, in_height // 4, in_width // 4, sphereType, 64, 128, 8, 1, 1, 1)  # sphere
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
    if block == SphereBasicBlock:
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
    output_feature = torch.cat((output_raw, output_regualr, output_sphere), 1)
    output_feature = self.lastconv(output_feature)

    return output_feature


# NOTE: feature extraction used in PSMNet
class feature_extraction(nn.Module):
  def __init__(self):
    super(feature_extraction, self).__init__()
    self.inplanes = 32
    self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))

    self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
    self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
    self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
    self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

    self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))

    self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))

    self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))

    self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))

    self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

  def _make_layer(self, block, planes, blocks, stride, pad, dilation):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
          nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

    return nn.Sequential(*layers)

  def forward(self, x):
    output = self.firstconv(x)
    output = self.layer1(output)
    output_raw = self.layer2(output)
    output = self.layer3(output_raw)
    output_skip = self.layer4(output)

    output_branch1 = self.branch1(output_skip)
    output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

    output_branch2 = self.branch2(output_skip)
    output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

    output_branch3 = self.branch3(output_skip)
    output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

    output_branch4 = self.branch4(output_skip)
    output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear', align_corners=True)

    output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
    output_feature = self.lastconv(output_feature)

    return output_feature