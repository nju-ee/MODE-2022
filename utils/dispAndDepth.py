import os
import sys
import torch
import numpy as np
import cv2
# delta_Theta = cos(Theta)/depth*baseline
# disp = delta_Theta/pi*h
# disp*depth = cos(Theta)*baseline*h/pi

# height, width = 256, 512
# minTheta = -np.pi / 2
# baseline = 0.26
# maxDepth = 20.0


def generateDispDepCoeMap(height=256, width=512, minTheta=-np.pi / 2, baseline=0.26):
  v = np.expand_dims(np.linspace(0, height - 1, height), 0).T
  v = np.repeat(v, width, 1)
  theta = v / height * np.pi + minTheta
  coe = (np.cos(theta) * baseline * height / np.pi).astype(np.float32)
  return coe


def batchConvertDispDepthERP(input, coeMap):
  b, c, h, w = input.shape
  coeMap = coeMap.repeat(b, c, 1, 1)
  output = coeMap / input
  return output


class DispDepthTransformerCassini():
  '''
  Disparity - Depth Transformer of Omnidirectional images in Cassini projection
  Based on the law of Sines
  r = baseline*(sin(phi)/tan(d)-cos(phi))
  d = arctan(sin(phi)/(r/baseline + cos(phi)))
  '''
  def __init__(self, height, width, baseline, maxDepth, cuda=True):
    self.width = width
    self.height = height
    self.phi2 = (np.repeat(np.expand_dims(np.linspace(0, self.width - 1, self.width), 0), self.height, 0) + 0.5) / self.width * np.pi
    cosPhi2 = np.cos(self.phi2).astype(np.float32)
    sinPhi2 = np.sin(self.phi2).astype(np.float32)
    self.b = baseline
    self.cuda = cuda
    self.cosPhi2 = torch.from_numpy(cosPhi2)
    self.sinPhi2 = torch.from_numpy(sinPhi2)
    self.minDispTh = 1e-10
    self.maxDepth = maxDepth
    if self.cuda:
      self.cosPhi2 = self.cosPhi2.cuda()
      self.sinPhi2 = self.sinPhi2.cuda()

  def disp2depth(self, disp):
    b, c, h, w = disp.shape
    cosPhi2 = self.cosPhi2.repeat(b, c, 1, 1)
    sinPhi2 = self.sinPhi2.repeat(b, c, 1, 1)
    mask0 = disp < self.minDispTh
    maskn0 = disp >= self.minDispTh
    depth = torch.zeros_like(disp)
    depth[maskn0] = self.b * (sinPhi2[maskn0] / torch.tan((disp[maskn0]) / self.width * np.pi) - cosPhi2[maskn0])
    depth[mask0] = self.maxDepth
    return depth

  def depth2disp(self, depth):
    b, c, h, w = depth.shape
    cosPhi2 = self.cosPhi2.repeat(b, c, 1, 1)
    sinPhi2 = self.sinPhi2.repeat(b, c, 1, 1)
    disp = torch.atan(sinPhi2 / (depth / self.b + cosPhi2))
    disp[disp < 0] += np.pi
    disp = (disp) / np.pi * self.width
    return disp


if __name__ == '__main__':
  sys.path.append('.')
  import ERPandCassini as EC

  maxDepth = 20.0
  name = "24_edb61af9bebd428aa21a59c4b2597b201"
  root = "/home/liming/Project/datasets/3D60/"
  leftImgName = os.path.join(root, "Center_Left_Down/Matterport3D/", name + "_color_0_Left_Down_0.0.png")
  rightImgName = os.path.join(root, "Right/Matterport3D/", name + "_color_0_Right_0.0.png")
  depthName = os.path.join(root, "Center_Left_Down/Matterport3D/", name + "_depth_0_Left_Down_0.0.exr")
  print(depthName)
  img = np.array(np.array(cv2.imread(depthName, cv2.IMREAD_ANYDEPTH)))
  img = torch.from_numpy(img)
  eh, ew = img.shape
  img = img.repeat(1, 1, 1, 1)
  invalidMask = img > maxDepth
  e2ca = EC.ERP2CA(eh, ew, ew, eh)
  ca2e = EC.CA2ERP(eh, ew, ew, eh)

# name = "24_edb61af9bebd428aa21a59c4b2597b201"
# root = "/home/liming/Project/datasets/3D60/"
# leftImgName = os.path.join(root, "Center_Left_Down/Matterport3D/", name + "_color_0_Left_Down_0.0.png")
# rightImgName = os.path.join(root, "Right/Matterport3D/", name + "_color_0_Right_0.0.png")
# depthName = os.path.join(root, "Center_Left_Down/Matterport3D/", name + "_depth_0_Left_Down_0.0.exr")
# print(depthName)
# img = np.array(np.array(cv2.imread(depthName, cv2.IMREAD_ANYDEPTH)))
# img = torch.from_numpy(img)
# img = img.repeat(1, 1, 1, 1)
# invalidMask = img > maxDepth
# coeMap = generateDispDepCoeMap()
# coeMap = torch.from_numpy(coeMap)
# print(coeMap.shape)
# disp = batchConvertDispDepthERP(img, coeMap)
# depth = batchConvertDispDepthERP(disp, coeMap)
# disp[invalidMask] = 0.0
# disp.squeeze_(0)
# disp.squeeze_(0)
# disp = disp.numpy()
# dispSave = ((disp - np.min(disp)) / (np.max(disp) - np.min(disp)) * 255).astype(np.int)
# depth[invalidMask] = 0.0
# depth.squeeze_(0)
# depth.squeeze_(0)
# depth = depth.numpy()
# depthSave = ((depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255).astype(np.int)
# cv2.imwrite('disp.png', dispSave)
# cv2.imwrite('depth.png', depthSave)
