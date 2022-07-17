from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


#------ Submodule ------#
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

  return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_planes))


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True))

    self.conv2 = nn.Sequential(convbn(planes, planes, 3, 1, pad, dilation), nn.ReLU(inplace=True))

    self.stride = stride

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    return out


class feature_extraction_Baseline(nn.Module):
  def __init__(self, maxdepth):
    super(feature_extraction_Baseline, self).__init__()
    self.inplanes = 6

    self.layer1 = self._make_layer(BasicBlock, 32, 2, 1, 1, 1)
    self.layer2 = self._make_layer(BasicBlock, 64, 1, 1, 1, 1)
    self.layer3 = self._make_layer(BasicBlock, 128, 1, 1, 1, 1)
    self.layer4 = self._make_layer(BasicBlock, 256, 1, 1, 1, 1)
    self.layer5 = self._make_layer(BasicBlock, 128, 1, 1, 1, 1)
    self.layer6 = self._make_layer(BasicBlock, 64, 1, 1, 1, 1)
    self.layer7 = self._make_last_layer(BasicBlock, 32, 2, 1, 1, 1)

    self.maxdepth = torch.tensor(maxdepth)

  def _make_layer(self, block, planes, blocks, stride, pad, dilation):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

    return nn.Sequential(*layers)

  def _make_last_layer(self, block, planes, blocks, stride, pad, dilation):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

    layers.append(nn.Conv2d(planes, 1, kernel_size=1, padding=0, stride=1, bias=True))
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


class feature_extraction_MODE_Fusion(nn.Module):
  def __init__(self, maxdepth, channels, inplanes):
    super(feature_extraction_MODE_Fusion, self).__init__()
    self.depth_inplanes = inplanes['depth']
    self.rgb_inplanes = inplanes['rgb']

    self.depth_layer1 = self._make_depth_layer(BasicBlock, channels[0], 2, 1, 1, 1)
    self.depth_layer2 = self._make_depth_layer_down(BasicBlock, channels[1], 1, 1, 1, 1)
    self.depth_layer3 = self._make_depth_layer_down(BasicBlock, channels[2], 1, 1, 1, 1)

    self.rgb_layer1 = self._make_rgb_layer(BasicBlock, channels[0], 2, 1, 1, 1)
    self.rgb_layer2 = self._make_rgb_layer_down(BasicBlock, channels[1], 1, 1, 1, 1)
    self.rgb_layer3 = self._make_rgb_layer_down(BasicBlock, channels[2], 1, 1, 1, 1)

    self.fusion_layer1 = self._make_fusion_layer(BasicBlock, channels[0], 2, 1, 1, 1)
    self.fusion_layer2 = self._make_fusion_layer(BasicBlock, channels[1], 2, 1, 1, 1)
    self.fusion_layer3 = self._make_fusion_layer(BasicBlock, channels[2], 2, 1, 1, 1)

    self.depth_layer4 = self._make_depth_layer_down_up(BasicBlock, channels[3], 1, 1, 1, 1)
    self.depth_layer5 = self._make_depth_layer_up(BasicBlock, channels[2], 1, 1, 1, 1)
    self.depth_layer6 = self._make_depth_layer_up(BasicBlock, channels[1], 1, 1, 1, 1)

    self.depth_layer7 = self._make_last_layer(BasicBlock, channels[0], 2, 1, 1, 1)

    self.maxdepth = torch.tensor(maxdepth)

  def _make_depth_layer(self, block, planes, blocks, stride, pad, dilation):
    downsample = None
    if stride != 1 or self.depth_inplanes != planes * block.expansion:
      downsample = nn.Sequential(nn.Conv2d(self.depth_inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

    layers = []
    layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
    self.depth_inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.depth_inplanes, planes, 1, None, pad, dilation))

    return nn.Sequential(*layers)

  def _make_rgb_layer(self, block, planes, blocks, stride, pad, dilation):
    downsample = None
    if stride != 1 or self.rgb_inplanes != planes * block.expansion:
      downsample = nn.Sequential(nn.Conv2d(self.rgb_inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

    layers = []
    layers.append(block(self.rgb_inplanes, planes, stride, downsample, pad, dilation))
    self.rgb_inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.rgb_inplanes, planes, 1, None, pad, dilation))

    return nn.Sequential(*layers)

  def _make_depth_layer_down(self, block, planes, blocks, stride, pad, dilation):
    downsample = None
    if stride != 1 or self.depth_inplanes != planes * block.expansion:
      downsample = nn.Sequential(nn.Conv2d(self.depth_inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

    layers = []
    layers.append(nn.MaxPool2d(2, stride=2))
    layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
    self.depth_inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.depth_inplanes, planes, 1, None, pad, dilation))

    return nn.Sequential(*layers)

  def _make_rgb_layer_down(self, block, planes, blocks, stride, pad, dilation):
    downsample = None
    if stride != 1 or self.rgb_inplanes != planes * block.expansion:
      downsample = nn.Sequential(nn.Conv2d(self.rgb_inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

    layers = []
    layers.append(nn.MaxPool2d(2, stride=2))
    layers.append(block(self.rgb_inplanes, planes, stride, downsample, pad, dilation))
    self.rgb_inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.rgb_inplanes, planes, 1, None, pad, dilation))

    return nn.Sequential(*layers)

  def _make_fusion_layer(self, block, planes, blocks, stride, pad, dilation):
    downsample = nn.Sequential(nn.Conv2d(int(2 * planes), planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

    layers = []
    layers.append(block(int(2 * planes), planes, stride, downsample, pad, dilation))
    for i in range(1, blocks):
      layers.append(block(planes, planes, 1, None, pad, dilation))

    return nn.Sequential(*layers)

  def _make_depth_layer_down_up(self, block, planes, blocks, stride, pad, dilation):
    downsample = None
    if stride != 1 or self.depth_inplanes != planes * block.expansion:
      downsample = nn.Sequential(nn.Conv2d(self.depth_inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

    layers = []
    layers.append(nn.MaxPool2d(2, stride=2))
    layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
    self.depth_inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.depth_inplanes, planes, 1, None, pad, dilation))
    layers.append(nn.ConvTranspose2d(planes, int(planes / 2), 2, 2))
    layers.append(nn.BatchNorm2d(int(planes / 2)))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

  def _make_depth_layer_up(self, block, planes, blocks, stride, pad, dilation):
    downsample = None
    if stride != 1 or self.depth_inplanes != planes * block.expansion:
      downsample = nn.Sequential(nn.Conv2d(self.depth_inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

    layers = []
    layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
    self.depth_inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.depth_inplanes, planes, 1, None, pad, dilation))
    layers.append(nn.ConvTranspose2d(planes, int(planes / 2), 2, 2))
    layers.append(nn.BatchNorm2d(int(planes / 2)))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

  def _make_last_layer(self, block, planes, blocks, stride, pad, dilation):
    downsample = None
    if stride != 1 or self.depth_inplanes != planes * block.expansion:
      downsample = nn.Sequential(nn.Conv2d(self.depth_inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

    layers = []
    layers.append(block(self.depth_inplanes, planes, stride, downsample, pad, dilation))
    self.depth_inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.depth_inplanes, planes, 1, None, pad, dilation))

    layers.append(nn.Conv2d(planes, 1, kernel_size=1, padding=0, stride=1, bias=True))
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
    self.lastconv = nn.Sequential(convbn(64, 32, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1, bias=True), nn.Sigmoid())
    self.maxdepth = torch.tensor(maxdepth)

  def forward(self, x):
    out = self.lastconv(x)
    return out * self.maxdepth


#------ Fusion Model ------#
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
    pred = self.feature_extraction(input_feature)
    return pred


class ModeFusion(nn.Module):
  def __init__(self, maxdepth, channels, inplanes):
    super(ModeFusion, self).__init__()
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

    pred = self.feature_extraction(depth_conf_input, rgb_input)

    return pred
