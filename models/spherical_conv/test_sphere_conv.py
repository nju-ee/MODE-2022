"""
NOTE:
test code of sphere convolution
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import cv2
import time
import random

import torchvision

from sphere_conv import SphereConv
from sphere_conv_pytorch import SphereConv2d


class Test3D60Dataset(Dataset):
  def __init__(self, rootDir, fileName):
    super(Test3D60Dataset, self).__init__()
    self.rootDir = rootDir
    self.fileName = fileName
    self.fileList = []
    with open(self.fileName) as f:
      lines = f.readlines()
      for l in lines:
        self.fileList.append(l.strip().split(" "))
    self.to_tensor = transforms.ToTensor()

  def __getitem__(self, index):
    names = self.fileList[index]
    rgbName = os.path.join(self.rootDir, names[0])
    depthName = os.path.join(self.rootDir, names[3])
    #print(rgbName, depthName)
    rgb = cv2.imread(rgbName)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    gtDepth = cv2.imread(depthName, cv2.IMREAD_ANYDEPTH)
    rgb = self.to_tensor(rgb)
    gtDepth = torch.from_numpy(gtDepth).unsqueeze_(0)
    data = {"rgb": rgb, "depth": gtDepth}
    return data

  def __len__(self):
    return len(self.fileList)


class TestSphereMultiLayer(nn.Module):
  def __init__(self, sphereType='ERP'):
    super(TestSphereMultiLayer, self).__init__()
    self.s1 = SphereConv(in_height=16, in_width=32, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, sphereType=sphereType)
    self.s2 = SphereConv(in_height=16, in_width=32, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, sphereType=sphereType)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.s1(x)
    out = self.s2(out)
    return self.relu(out)


class TestSphereDebugModel(nn.Module):
  def __init__(self, in_height, in_width, proType='ERP', F=16, max_depth=20.0, modelType='Sphere'):
    super(TestSphereDebugModel, self).__init__()
    self.in_height, self.in_width = in_height, in_width
    self.proType = proType
    self.maxDepth = max_depth
    #h_in, w_in = self.in_height, in_width
    self.fea = F
    self.conv1 = SphereConv(in_height=self.in_height,
                            in_width=self.in_width,
                            in_channels=3,
                            out_channels=self.fea,
                            kernel_size=7,
                            stride=1,
                            padding=3,
                            dilation=1,
                            groups=1,
                            bias=False,
                            sphereType=self.proType)
    self.bn1 = nn.BatchNorm2d(self.fea)
    self.ac1 = nn.ReLU(inplace=True)

  def forward(self, x):
    print(x.shape)
    out = self.conv1(x)
    print(out.shape)
    out = self.bn1(out)
    out = self.ac1(out)


class TestSphereModel(nn.Module):
  def __init__(self, in_height, in_width, proType='ERP', F=16, max_depth=20.0, modelType='Sphere'):
    super(TestSphereModel, self).__init__()
    self.in_height, self.in_width = in_height, in_width
    self.proType = proType
    self.maxDepth = max_depth
    #h_in, w_in = self.in_height, in_width
    self.fea = F
    if modelType == 'Sphere':
      self.__buildSphereModel()
    elif modelType == 'Regular':
      self.__buildRegularModel()
    elif modelType == 'SpherePy':
      self.__buildSpherePyModel()
    else:
      raise NotImplementedError("model type must be Sphere or Regular!")

  def __buildSphereModel(self):
    self.inConv = nn.Sequential(
        SphereConv(in_height=self.in_height,
                   in_width=self.in_width,
                   in_channels=3,
                   out_channels=self.fea,
                   kernel_size=7,
                   stride=1,
                   padding=3,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.BatchNorm2d(self.fea),
        nn.ReLU(inplace=True))
    self.sd1 = nn.Sequential(
        SphereConv(in_height=self.in_height,
                   in_width=self.in_width,
                   in_channels=self.fea,
                   out_channels=2 * self.fea,
                   kernel_size=3,
                   stride=2,
                   padding=1,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.BatchNorm2d(2 * self.fea),
        nn.ReLU(inplace=True),
        SphereConv(in_height=self.in_height // 2,
                   in_width=self.in_width // 2,
                   in_channels=2 * self.fea,
                   out_channels=2 * self.fea,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.BatchNorm2d(2 * self.fea),
        nn.ReLU(inplace=True))
    self.sd2 = nn.Sequential(
        SphereConv(in_height=self.in_height // 2,
                   in_width=self.in_width // 2,
                   in_channels=2 * self.fea,
                   out_channels=4 * self.fea,
                   kernel_size=3,
                   stride=2,
                   padding=1,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.BatchNorm2d(4 * self.fea),
        nn.ReLU(inplace=True),
        SphereConv(in_height=self.in_height // 4,
                   in_width=self.in_width // 4,
                   in_channels=4 * self.fea,
                   out_channels=4 * self.fea,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.BatchNorm2d(4 * self.fea),
        nn.ReLU(inplace=True))
    self.ss = nn.Sequential(
        SphereConv(in_height=self.in_height // 4,
                   in_width=self.in_width // 4,
                   in_channels=4 * self.fea,
                   out_channels=8 * self.fea,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.BatchNorm2d(8 * self.fea),
        nn.ReLU(inplace=True),
        SphereConv(in_height=self.in_height // 4,
                   in_width=self.in_width // 4,
                   in_channels=8 * self.fea,
                   out_channels=4 * self.fea,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.BatchNorm2d(4 * self.fea),
        nn.ReLU(inplace=True))
    self.su1 = nn.Sequential(
        SphereConv(in_height=self.in_height // 2,
                   in_width=self.in_width // 2,
                   in_channels=4 * self.fea,
                   out_channels=2 * self.fea,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.BatchNorm2d(2 * self.fea),
        nn.ReLU(inplace=True),
        SphereConv(in_height=self.in_height // 2,
                   in_width=self.in_width // 2,
                   in_channels=2 * self.fea,
                   out_channels=2 * self.fea,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.BatchNorm2d(2 * self.fea),
        nn.ReLU(inplace=True))
    self.su2 = nn.Sequential(
        SphereConv(in_height=self.in_height,
                   in_width=self.in_width,
                   in_channels=2 * self.fea,
                   out_channels=self.fea,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.BatchNorm2d(self.fea),
        nn.ReLU(inplace=True),
        SphereConv(in_height=self.in_height,
                   in_width=self.in_width,
                   in_channels=self.fea,
                   out_channels=self.fea,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.BatchNorm2d(self.fea),
        nn.ReLU(inplace=True))
    self.outConv = nn.Sequential(
        SphereConv(in_height=self.in_height,
                   in_width=self.in_width,
                   in_channels=self.fea,
                   out_channels=1,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   dilation=1,
                   groups=1,
                   bias=False,
                   sphereType=self.proType),
        nn.Sigmoid())

  def __buildRegularModel(self):
    self.inConv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=self.fea, kernel_size=7, stride=1, padding=3, dilation=1, groups=1, bias=False), nn.BatchNorm2d(self.fea), nn.ReLU(inplace=True))
    self.sd1 = nn.Sequential(nn.Conv2d(in_channels=self.fea,
                                       out_channels=2 * self.fea,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       bias=False),
                             nn.BatchNorm2d(2 * self.fea),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channels=2 * self.fea,
                                       out_channels=2 * self.fea,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       bias=False),
                             nn.BatchNorm2d(2 * self.fea),
                             nn.ReLU(inplace=True))
    self.sd2 = nn.Sequential(nn.Conv2d(in_channels=2 * self.fea,
                                       out_channels=4 * self.fea,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       bias=False),
                             nn.BatchNorm2d(4 * self.fea),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channels=4 * self.fea,
                                       out_channels=4 * self.fea,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       bias=False),
                             nn.BatchNorm2d(4 * self.fea),
                             nn.ReLU(inplace=True))
    self.ss = nn.Sequential(nn.Conv2d(in_channels=4 * self.fea,
                                      out_channels=8 * self.fea,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      dilation=1,
                                      groups=1,
                                      bias=False),
                            nn.BatchNorm2d(8 * self.fea),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=8 * self.fea,
                                      out_channels=4 * self.fea,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      dilation=1,
                                      groups=1,
                                      bias=False),
                            nn.BatchNorm2d(4 * self.fea),
                            nn.ReLU(inplace=True))
    self.su1 = nn.Sequential(nn.Conv2d(in_channels=4 * self.fea,
                                       out_channels=2 * self.fea,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       bias=False),
                             nn.BatchNorm2d(2 * self.fea),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channels=2 * self.fea,
                                       out_channels=2 * self.fea,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       bias=False),
                             nn.BatchNorm2d(2 * self.fea),
                             nn.ReLU(inplace=True))
    self.su2 = nn.Sequential(nn.Conv2d(in_channels=2 * self.fea,
                                       out_channels=self.fea,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       bias=False),
                             nn.BatchNorm2d(self.fea),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channels=self.fea,
                                       out_channels=self.fea,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       dilation=1,
                                       groups=1,
                                       bias=False),
                             nn.BatchNorm2d(self.fea),
                             nn.ReLU(inplace=True))
    self.outConv = nn.Sequential(nn.Conv2d(in_channels=self.fea, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False), nn.Sigmoid())

  def __buildSpherePyModel(self):
    self.inConv = nn.Sequential(
        SphereConv2d(in_height=self.in_height,
                     in_width=self.in_width,
                     in_channels=3,
                     out_channels=self.fea,
                     kernel_size=7,
                     stride=1,
                     padding=3,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.BatchNorm2d(self.fea),
        nn.ReLU(inplace=True))
    self.sd1 = nn.Sequential(
        SphereConv2d(in_height=self.in_height,
                     in_width=self.in_width,
                     in_channels=self.fea,
                     out_channels=2 * self.fea,
                     kernel_size=3,
                     stride=2,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.BatchNorm2d(2 * self.fea),
        nn.ReLU(inplace=True),
        SphereConv2d(in_height=self.in_height // 2,
                     in_width=self.in_width // 2,
                     in_channels=2 * self.fea,
                     out_channels=2 * self.fea,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.BatchNorm2d(2 * self.fea),
        nn.ReLU(inplace=True))
    self.sd2 = nn.Sequential(
        SphereConv2d(in_height=self.in_height // 2,
                     in_width=self.in_width // 2,
                     in_channels=2 * self.fea,
                     out_channels=4 * self.fea,
                     kernel_size=3,
                     stride=2,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.BatchNorm2d(4 * self.fea),
        nn.ReLU(inplace=True),
        SphereConv2d(in_height=self.in_height // 4,
                     in_width=self.in_width // 4,
                     in_channels=4 * self.fea,
                     out_channels=4 * self.fea,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.BatchNorm2d(4 * self.fea),
        nn.ReLU(inplace=True))
    self.ss = nn.Sequential(
        SphereConv2d(in_height=self.in_height // 4,
                     in_width=self.in_width // 4,
                     in_channels=4 * self.fea,
                     out_channels=8 * self.fea,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.BatchNorm2d(8 * self.fea),
        nn.ReLU(inplace=True),
        SphereConv2d(in_height=self.in_height // 4,
                     in_width=self.in_width // 4,
                     in_channels=8 * self.fea,
                     out_channels=4 * self.fea,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.BatchNorm2d(4 * self.fea),
        nn.ReLU(inplace=True))
    self.su1 = nn.Sequential(
        SphereConv2d(in_height=self.in_height // 2,
                     in_width=self.in_width // 2,
                     in_channels=4 * self.fea,
                     out_channels=2 * self.fea,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.BatchNorm2d(2 * self.fea),
        nn.ReLU(inplace=True),
        SphereConv2d(in_height=self.in_height // 2,
                     in_width=self.in_width // 2,
                     in_channels=2 * self.fea,
                     out_channels=2 * self.fea,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.BatchNorm2d(2 * self.fea),
        nn.ReLU(inplace=True))
    self.su2 = nn.Sequential(
        SphereConv2d(in_height=self.in_height,
                     in_width=self.in_width,
                     in_channels=2 * self.fea,
                     out_channels=self.fea,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.BatchNorm2d(self.fea),
        nn.ReLU(inplace=True),
        SphereConv2d(in_height=self.in_height,
                     in_width=self.in_width,
                     in_channels=self.fea,
                     out_channels=self.fea,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.BatchNorm2d(self.fea),
        nn.ReLU(inplace=True))
    self.outConv = nn.Sequential(
        SphereConv2d(in_height=self.in_height,
                     in_width=self.in_width,
                     in_channels=self.fea,
                     out_channels=1,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=1,
                     bias=False),
        nn.Sigmoid())

  def forward(self, x):
    out = self.inConv(x)
    print(out.shape)
    out = self.sd1(out)
    out = self.sd2(out)
    out = self.ss(out)
    out = F.interpolate(out, scale_factor=2.0)
    out = self.su1(out)
    out = F.interpolate(out, scale_factor=2.0)
    out = self.su2(out)
    out = self.outConv(out)
    return self.maxDepth * out


def myBilinear(x, height, width, p_h, p_w):
  h_low = int(np.floor(p_h))
  h_high = h_low + 1
  w_low = int(np.floor(p_w))
  w_high = w_low + 1
  lh = p_h - h_low
  lw = p_w - w_low
  hh = 1 - lh
  hw = 1 - lw
  v1 = 0
  if (h_low >= 0 and w_low >= 0):
    v1 = x[:, :, h_low, w_low]
  v2 = 0
  if (h_low >= 0 and w_high <= width - 1):
    v2 = x[:, :, h_low, w_high]
  v3 = 0
  if (h_high <= height - 1 and w_low >= 0):
    v3 = x[:, :, h_high, w_low]
  v4 = 0
  if (h_high <= height - 1 and w_high <= width - 1):
    v4 = x[:, :, h_high, w_high]

  w1 = hh * hw
  w2 = hh * lw
  w3 = lh * hw
  w4 = lh * lw

  val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)
  return val


def positionTest():
  # position testing
  #torch.backends.cudnn.enable = True
  proType = 'Cassini'
  if proType == 'ERP':
    h_in, w_in = 16, 32
  elif proType == 'Cassini':
    h_in, w_in = 32, 16
  else:
    print("Projection Type Error. Only accept ERP or Cassini projection")
  kk = 3
  ss = 1
  pp = kk // 2
  h_out = (h_in + 2 * pp - kk) // ss + 1
  w_out = (w_in + 2 * pp - kk) // ss + 1
  tsn = SphereConv(in_height=h_in, in_width=w_in, in_channels=1, out_channels=1, kernel_size=kk, stride=ss, padding=pp, dilation=1, groups=1, bias=False, sphereType=proType).cuda()
  target = torch.ones((1, 1, h_out, w_out)).cuda()
  opt = torch.optim.SGD(tsn.parameters(), lr=0.1)
  pos = tsn.getPosition()
  x = torch.randn((1, 1, h_in, w_in)).float().cuda()
  print("input: \n", x.shape)
  #np.save('input.npy', x.cpu().numpy())
  print("position: \n", pos.shape)
  #np.save("pos.npy", pos.cpu().numpy())
  print("weights: ")
  weight = None
  for k, v in tsn.state_dict().items():
    print(k, v)
    weight = v
  print(weight)
  ho, wo = 5, 3
  hc = ho * ss
  wc = wo * ss
  y = tsn(x)
  vals = []
  for i in range(kk * kk):
    p_h = pos[:, i * 2, hc, wc].data.item()
    p_w = pos[:, i * 2 + 1, hc, wc].data.item()
    val = myBilinear(x, h_in, w_in, p_h, p_w)
    print(p_h, p_w, val)
    vals.append(val)
  vals = torch.cat(vals, dim=1).unsqueeze_(0).unsqueeze_(0).view(1, 1, kk, kk)
  out0 = torch.sum(weight * vals)
  print(out0)
  print(vals)

  print(y[:, :, ho, wo])
  print("output: \n", y.shape)
  #np.save('output.npy', y.detach().cpu().numpy())
  loss = F.smooth_l1_loss(y, target)
  loss.backward()
  opt.step()
  for k, v in tsn.state_dict().items():
    print(k, v)


def multiLayerTest():
  # multi layer forward & backward testing
  tsn = TestSphereMultiLayer('ERP').cuda()
  x = torch.randn((1, 1, 16, 32)).float().cuda()
  t = torch.ones_like(x)
  opt = torch.optim.SGD(tsn.parameters(), lr=0.01)
  tsn.train()
  opt.zero_grad()
  print("x: \n", x)
  print("t: \n", t)
  i = 0
  print("spc:\n")
  for k, v in tsn.state_dict().items():
    print(k, v)
  while (i < 10):
    out = tsn(x)
    loss = F.smooth_l1_loss(out, t)
    print(loss)
    loss.backward()
    opt.step()
    print("spc:\n")
    for k, v in tsn.state_dict().items():
      print(k, v)
    i = i + 1


def testFullSphereModel():
  seed = 111
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  modelType = 'Sphere'
  model = TestSphereDebugModel(in_height=256, in_width=512, modelType=modelType)
  model = model.cuda()
  model = nn.DataParallel(model)
  opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
  trData = Test3D60Dataset('/home/liming/projects/datasets/3D60/Center_Left_Down/', './train.txt')
  vaData = Test3D60Dataset('/home/liming/projects/datasets/3D60/Center_Left_Down/', './val.txt')
  trLoader = DataLoader(trData, batch_size=2, shuffle=True, num_workers=4, pin_memory=False, drop_last=False)
  vaLoader = DataLoader(vaData, batch_size=2, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)
  maxEpoch = 30
  maxDepth = 20.0
  print("start!")
  for e in range(maxEpoch):
    startTime = time.time()
    totalTrainLoss = 0.0
    count = 0
    model.train()
    for batchId, batchData in enumerate(trLoader):
      opt.zero_grad()
      rgb = batchData['rgb'].cuda()
      depth = batchData['depth'].cuda()
      mask = (depth <= maxDepth) & (depth > 0)
      t1 = time.time()
      pred = model(rgb)
      t2 = time.time()
      loss = F.smooth_l1_loss(pred[mask], depth[mask])
      loss.backward()
      opt.step()
      t3 = time.time()
      totalTrainLoss += loss
      count += 1
      print("forward: {}. backward: {}".format(t2 - t1, t3 - t2))

    print("Training. Epoch: {}. Loss: {}. Time: {}".format(e, totalTrainLoss / count, time.time() - startTime))
    startTime = time.time()
    totalTrainLoss = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
      for batchId, batchData in enumerate(vaLoader):
        rgb = batchData['rgb'].cuda()
        depth = batchData['depth'].cuda()
        mask = (depth <= maxDepth) & (depth > 0)
        pred = model(rgb)
        loss = F.smooth_l1_loss(pred[mask], depth[mask])
        totalTrainLoss += loss
        count += 1
        if batchId == 0:
          saveImg = pred[0, ::]
          saveImg = (saveImg - torch.min(saveImg)) / (torch.max(saveImg) - torch.min(saveImg))
          torchvision.utils.save_image(saveImg, '{}_val_{}.png'.format(modelType, e))
    print("Validation. Epoch: {}. Loss: {}. Time: {}".format(e, totalTrainLoss / count, time.time() - startTime))


if __name__ == '__main__':
  # positionTest()
  # multiLayerTest()
  testFullSphereModel()
