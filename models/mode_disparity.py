from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *


class hourglass(nn.Module):
  def __init__(self, inplanes):
    super(hourglass, self).__init__()

    self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))

    self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

    self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))

    self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1), nn.ReLU(inplace=True))

    self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(inplanes * 2))  #+conv2

    self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(inplanes))  #+x

  def forward(self, x, presqu, postsqu):

    out = self.conv1(x)  #in:1/4 out:1/8
    pre = self.conv2(out)  #in:1/8 out:1/8
    if postsqu is not None:
      pre = F.relu(pre + postsqu, inplace=True)
    else:
      pre = F.relu(pre, inplace=True)

    out = self.conv3(pre)  #in:1/8 out:1/16
    out = self.conv4(out)  #in:1/16 out:1/16

    if presqu is not None:
      post = F.relu(self.conv5(out) + presqu, inplace=True)  #in:1/16 out:1/8
    else:
      post = F.relu(self.conv5(out) + pre, inplace=True)

    out = self.conv6(post)  #in:1/8 out:1/4

    return out, pre, post


# Note
# in_height,in_width: shape of input image. using (1024,512) for deep360, (640,320) for fisheye, (512,256) for 3D60
class ModeDisparity(nn.Module):
  def __init__(self, maxdisp, conv='Sphere', in_height=1024, in_width=512, sphereType='Cassini', out_conf=False):
    super(ModeDisparity, self).__init__()
    print("MODE stereo matching network!")
    self.maxdisp = maxdisp
    self.out_conf = out_conf
    if conv == 'Regular':
      self.feature_extraction = feature_extraction()
      print("using Regular feature extraction!")
    elif conv == 'Sphere':
      self.feature_extraction = sphere_feature_extraction(in_height, in_width, sphereType)
      print("using Spherical feature extraction!")
    else:
      raise NotImplementedError("Convolution Type must be Regular or Sphere!")

    self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))

    self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1))

    self.dres2 = hourglass(32)

    self.dres3 = hourglass(32)

    self.dres4 = hourglass(32)

    self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.Conv3d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

  def forward(self, left, right):

    refimg_fea = self.feature_extraction(left)
    targetimg_fea = self.feature_extraction(right)

    #matching
    cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp // 4, refimg_fea.size()[2], refimg_fea.size()[3]).zero_()).cuda()

    for i in range(self.maxdisp // 4):
      if i > 0:
        cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
        cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
      else:
        cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
        cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
    cost = cost.contiguous()

    cost0 = self.dres0(cost)
    cost0 = self.dres1(cost0) + cost0

    out1, pre1, post1 = self.dres2(cost0, None, None)
    out1 = out1 + cost0

    out2, pre2, post2 = self.dres3(out1, pre1, post1)
    out2 = out2 + cost0

    out3, pre3, post3 = self.dres4(out2, pre1, post2)
    out3 = out3 + cost0

    cost1 = self.classif1(out1)
    cost2 = self.classif2(out2) + cost1
    cost3 = self.classif3(out3) + cost2

    if self.training:
      cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
      cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)

      cost1 = torch.squeeze(cost1, 1)
      pred1 = F.softmax(cost1, dim=1)
      pred1 = disparityregression(self.maxdisp)(pred1)

      cost2 = torch.squeeze(cost2, 1)
      pred2 = F.softmax(cost2, dim=1)
      pred2 = disparityregression(self.maxdisp)(pred2)

    cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)
    cost3 = torch.squeeze(cost3, 1)
    pred3 = F.softmax(cost3, dim=1)
    if self.out_conf:
      prob_volume = pred3

    #For your information: This formulation 'softmax(c)' learned "similarity"
    #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
    #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
    pred3 = disparityregression(self.maxdisp)(pred3)

    if self.training:
      return pred1, pred2, pred3
    else:
      if self.out_conf:
        # Output Confidence Map
        grid_d = torch.round(pred3).permute([0, 2, 3, 1]).unsqueeze(1) / (self.maxdisp - 1.0) * 2 - 1
        grid_d_floor = (torch.round(pred3) - 1).permute([0, 2, 3, 1]).unsqueeze(1) / (self.maxdisp - 1) * 2 - 1
        grid_d_ceil = (torch.round(pred3) + 1).permute([0, 2, 3, 1]).unsqueeze(1) / (self.maxdisp - 1) * 2 - 1
        grid_h, grid_w = torch.meshgrid(torch.arange(0, grid_d.shape[2]), torch.arange(0, grid_d.shape[3]))
        grid_h = (grid_h / (grid_d.shape[2] - 1.0) * 2 - 1).unsqueeze(0).unsqueeze(-1).unsqueeze(0).repeat_interleave(grid_d.shape[0], dim=0).cuda()
        grid_w = (grid_w / (grid_d.shape[3] - 1.0) * 2 - 1).unsqueeze(0).unsqueeze(-1).unsqueeze(0).repeat_interleave(grid_d.shape[0], dim=0).cuda()
        grid = torch.cat([grid_w, grid_h, grid_d], dim=-1)
        grid_floor = torch.cat([grid_w, grid_h, grid_d_floor], dim=-1)
        grid_ceil = torch.cat([grid_w, grid_h, grid_d_ceil], dim=-1)
        prob_map = F.grid_sample(prob_volume.unsqueeze(1),
                                 grid,
                                 align_corners=True,
                                 padding_mode='border',
                                 mode='nearest') + F.grid_sample(prob_volume.unsqueeze(1),
                                                                 grid_floor,
                                                                 align_corners=True,
                                                                 padding_mode='border',
                                                                 mode='nearest') + F.grid_sample(prob_volume.unsqueeze(1),
                                                                                                 grid_ceil,
                                                                                                 align_corners=True,
                                                                                                 padding_mode='border',
                                                                                                 mode='nearest')
        prob_map = prob_map.squeeze(1)

        return pred3, prob_map
      else:
        return pred3
