import os
import sys
import cv2
import numpy as np
import math
import torch
from torch.autograd import Variable

# root = "../datasets/3D60"
# leftDir = "Center_Left_Down/Matterport3D/"
# rightDir = "Right/Matterport3D/"
# erpName = "0_0b217f59904d4bdf85d35da2cab963471_color_0_"
# erpImgLeft = cv2.imread(os.path.join(root, leftDir, erpName + "Left_Down_0.0.png"), cv2.IMREAD_ANYCOLOR)
# erpImgRight = cv2.imread(os.path.join(root, rightDir, erpName + "Right_0.0.png"), cv2.IMREAD_ANYCOLOR)
'''
Default Camera locations:

               front
        |--(1)------(2)--|
        |                |
  left  |    top view    |  right
        |                |
        |--(3)------(4)--|
               back
ERP and Cassini Trans needs a parameter phiBias, which determinds the bias between ERP and Cassini projection front directions. 
'''


class ERP2CA():
  '''
  A Class to transform left-right stereo ERP image pairs to Cassini image pairs
  Usage:
  e2ca = ERP2CA(heightE, widthE, heightC, widthC)
  ca12_1, ca12_2 = e2ca.trans(erpImg_1, erpImg_2, angle)
  '''
  def __init__(self, heightE, widthE, heightC, widthC, cuda=True):
    '''
    heightE, widthE: height and width of ERP images
    heightC, widthC: height and width of Cassini images
    Generate self.gridDict, the sampling grids of different camera pairs
    angle: the angle between the optical center of two cameras and the horizontal line
    (1)-(2):0 [(1) is left camera, and (2) is right camera]
    (1)-(3):90 [(1) is left camera, and (3) is right camera]
    (1)-(4):45 [(1) is left camera, and (4) is right camera]
    '''
    self.angleV = ['0', '45', '90', '135']
    self.gridDict = {'0': None, '45': None, '90': None, '135': None}
    self.phiBias = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    self.cuda = cuda
    phi2 = np.pi / 2 - np.repeat(np.expand_dims(np.linspace(0, widthC, widthC), 0), heightC, 0) / widthC * np.pi
    theta2 = np.repeat((np.expand_dims(np.linspace(0, heightC, heightC), 0).T), widthC, 1) / widthC * np.pi - np.pi / 2
    x = np.sin(phi2)
    y = np.cos(phi2) * np.cos(theta2)
    z = np.cos(phi2) * np.sin(theta2)

    for i in range(4):
      yaw = self.phiBias[i]
      Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, 1, 0], [np.sin(yaw), 0, np.cos(yaw)]])
      #Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
      X_2 = np.expand_dims(np.dstack((x, y, z)), axis=-1)
      X = np.matmul(Ry, X_2)
      phi = -np.arctan2(X[:, :, 0, 0], X[:, :, 2, 0]).astype(np.float32)
      theta = -np.arcsin(np.clip(X[:, :, 1, 0], -1, 1)).astype(np.float32)
      u = ((phi + np.pi) / np.pi * heightE) + widthE
      v = ((theta + np.pi / 2) / np.pi * heightE) + heightE
      u[u > widthE - 1] = u[u > widthE - 1] - (widthE - 1)
      v[v > heightE - 1] = v[v > heightE - 1] - (heightE - 1)
      u_ = (u - np.min(u)) / (np.max(u) - np.min(u)) * 2 - 1
      v_ = (v - np.min(u)) / (np.max(v) - np.min(v)) * 2 - 1
      u_ = torch.from_numpy(u_)
      v_ = torch.from_numpy(v_)
      u_.unsqueeze_(2)
      v_.unsqueeze_(2)
      self.gridDict[self.angleV[i]] = torch.cat([u_, v_], 2)

  def trans(self, img, angle):
    if not angle in self.angleV:
      print("transfer angle error, angle must be one of '0','45' and '90'! ")
      return None
    grid = self.gridDict[angle]
    if (self.cuda): grid = grid.cuda()
    b = img.shape[0]
    grid = grid.repeat(b, 1, 1, 1).float()
    caImg = torch.nn.functional.grid_sample(img, grid, align_corners=True)
    return caImg

  def transPairs(self, erpLeft, erpRight, angle):
    if not angle in self.angleV:
      print("transfer angle error, angle must be one of '0','45' and '90'! ")
      return None
    grid = self.gridDict[angle]
    b = erpLeft.shape[0]
    grid = grid.repeat(b, 1, 1, 1)
    caLeft = torch.nn.functional.grid_sample(erpLeft, grid, align_corners=True)
    caRight = torch.nn.functional.grid_sample(erpRight, grid, align_corners=True)
    return caLeft, caRight


class CA2ERP():
  '''
  A Class to transform left-right stereo ERP image pairs to Cassini image pairs
  Usage:
  ca2e = CA2ERP(heightE, widthE, heightC, widthC)
  erp12_1, erp12_2 = ca2e.trans(ca12_1, ca12_2, angle)
  '''
  def __init__(self, heightE, widthE, heightC, widthC, cuda=True, transGT=False):
    '''
    heightE, widthE: height and width of ERP images
    heightC, widthC: height and width of Cassini images
    Generate self.gridDict, the sampling grids of different camera pairs
    angle: the angle between the optical center of two cameras and the horizontal line
    (1)-(2):0 [(1) is left camera, and (2) is right camera]
    (1)-(3):90 [(1) is left camera, and (3) is right camera]
    (1)-(4):45 [(1) is left camera, and (4) is right camera]
    '''
    self.angleV = ['0', '45', '90', '135']
    self.gridDict = {'0': None, '45': None, '90': None, '135': None}
    self.phiBias = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    # to trans gt depth map to different view, use the opposite bias
    if transGT:
      self.phiBias = [0, -np.pi / 4, -np.pi / 2, -3 * np.pi / 4]
    self.cuda = cuda
    theta = np.pi / 2 - np.repeat((np.expand_dims(np.linspace(0, heightE, heightE), 0).T), widthE, 1) / heightE * np.pi
    phi = np.repeat(np.expand_dims(np.linspace(0, widthE, widthE), 0), heightE, 0) / heightE * np.pi - np.pi / 2
    x = np.cos(theta) * np.cos(phi)
    z = np.cos(theta) * np.sin(phi)
    y = np.sin(theta)
    for i in range(4):
      yaw = self.phiBias[i]
      Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, 1, 0], [np.sin(yaw), 0, np.cos(yaw)]])
      #Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
      X_2 = np.expand_dims(np.dstack((x, y, z)), axis=-1)
      X = np.matmul(np.linalg.inv(Ry), X_2)
      phi2 = -np.arcsin(np.clip(X[:, :, 0, 0], -1, 1)).astype(np.float32)
      theta2 = -np.arctan2(X[:, :, 1, 0], X[:, :, 2, 0]).astype(np.float32)
      u = ((phi2 + np.pi / 2) / np.pi * heightE) + widthC
      v = ((theta2 + np.pi) / np.pi * heightE) + heightC
      u[u > widthC - 1] = u[u > widthC - 1] - (widthC - 1)
      v[v > heightC - 1] = v[v > heightC - 1] - (heightC - 1)
      u_ = (u - np.min(u)) / (np.max(u) - np.min(u)) * 2 - 1
      v_ = (v - np.min(u)) / (np.max(v) - np.min(v)) * 2 - 1
      u_ = torch.from_numpy(u_)
      v_ = torch.from_numpy(v_)
      u_.unsqueeze_(2)
      v_.unsqueeze_(2)
      self.gridDict[self.angleV[i]] = torch.cat([u_, v_], 2)

  def trans(self, img, angle):
    if not angle in self.angleV:
      print("transfer angle error, angle must be one of '0','45' and '90'! ")
      return None
    grid = self.gridDict[angle]
    if (self.cuda): grid = grid.cuda()
    b = img.shape[0]
    grid = grid.repeat(b, 1, 1, 1).float()
    erpImg = torch.nn.functional.grid_sample(img, grid, align_corners=True)
    return erpImg

  def transPairs(self, caLeft, caRight, angle):
    if not angle in self.angleV:
      print("transfer angle error, angle must be one of '0','45' and '90'! ")
      return None
    grid = self.gridDict[angle]
    b = caLeft.shape[0]
    grid = grid.repeat(b, 1, 1, 1)
    erpLeft = torch.nn.functional.grid_sample(caLeft, grid, align_corners=True)
    erpRight = torch.nn.functional.grid_sample(caRight, grid, align_corners=True)
    return erpLeft, erpRight


def ERP2ERP(img, pitch, yaw, roll):
  Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
  Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
  Ry = np.array([[math.cos(pitch), 0, -math.sin(pitch)], [0, 1, 0], [math.sin(pitch), 0, math.cos(pitch)]])
  R = np.dot(np.dot(Rx, Rz), Ry)
  height, width = img.shape[-2:]
  theta = np.repeat((np.expand_dims(np.linspace(0, height, height), 0).T), width, 1) / height * np.pi - np.pi / 2
  phi = np.repeat(np.expand_dims(np.linspace(0, width, width), 0), height, 0) / height * np.pi - np.pi / 2
  x_2 = np.cos(theta) * np.cos(phi)
  z_2 = np.cos(theta) * np.sin(phi)
  y_2 = np.sin(theta)
  X_2 = np.expand_dims(np.dstack((x_2, y_2, z_2)), axis=-1)
  X = np.matmul(R, X_2)
  phi2 = np.arctan2(X[:, :, 2, 0], X[:, :, 0, 0]).astype(np.float32)
  theta2 = np.arcsin(np.clip(X[:, :, 1, 0], -1, 1)).astype(np.float32)
  print(np.max(phi2), np.min(phi2))

  source_image = torch.from_numpy(img).unsqueeze_(0)
  grid_x = Variable(torch.FloatTensor(np.clip(phi2 / (np.pi), -1, 1)).unsqueeze(-1))
  grid_y = Variable(torch.FloatTensor(np.clip(theta2 / (np.pi / 2), -1, 1)).unsqueeze(-1))

  grid = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0)
  sampled_image = torch.nn.functional.grid_sample(source_image, grid, mode='bilinear', align_corners=True, padding_mode='border')  # 1, ch, self.output_h, self.output_w

  cassini_2 = sampled_image.transpose(1, 3).transpose(1, 2).data.cpu().numpy()[0].astype(np.float32)
  return cassini_2


def ErpRoll(imghwc, angle):
  h, w, c = imghwc.shape
  if angle < 0:
    angle = 2 * np.pi - angle
  rollIdx = int(w * angle // (2 * np.pi))
  imgroll = np.roll(imghwc, rollIdx, axis=1)
  return imgroll


def ERP2Cassini(erpImgLeft, erpImgRight):
  print(erpImgLeft.shape)
  heightE, widthE = erpImgLeft.shape[0], erpImgLeft.shape[1]
  heightC, widthC = widthE, heightE
  phi2 = np.repeat(np.expand_dims(np.linspace(0, widthC - 1, widthC), 0), heightC, 0) / widthC * np.pi - np.pi / 2
  theta2 = -np.repeat((np.expand_dims(np.linspace(0, heightC - 1, heightC), 0).T), widthC, 1) / widthC * np.pi
  print(phi2)
  print(theta2)
  caLeft = np.ndarray([heightC, widthC, 3])
  caRight = np.ndarray([heightC, widthC, 3])
  x = np.sin(phi2)
  y = np.cos(phi2) * np.cos(theta2)
  z = np.cos(phi2) * np.sin(theta2)
  phi = np.arctan2(y, x)
  theta = np.arcsin(z)
  print(np.max(phi), np.min(phi))
  print(np.max(theta), np.min(theta))
  u = (((phi - np.pi / 4) / np.pi * heightE).astype(np.int) + widthE) % widthE  #phi+0:1/3; phi-np.pi/2:1/2
  print(np.max(u), np.min(u))
  v = (((theta + np.pi / 2) / np.pi * heightE).astype(np.int) + heightE) % heightE
  u[u > widthE - 1] = widthE - 1
  v[v > heightE - 1] = heightE - 1
  for i in range(heightC):
    for j in range(widthC):
      ii = u[i, j]
      jj = v[i, j]
      caLeft[i, j, :] = erpImgLeft[jj, ii, :]
      caRight[i, j, :] = erpImgRight[jj, ii, :]
  print(caLeft.shape)
  print(caLeft.dtype)
  print(np.max(caLeft), np.min(caLeft))
  cv2.imwrite('caleft.png', caLeft)
  cv2.imwrite('caright.png', caRight)
  return caLeft, caRight


def Cassini2ERP(caleft, caright):
  print(caleft.shape)
  heightC, widthC = caleft.shape[0], caleft.shape[1]
  heightE, widthE = widthC, heightC
  phi = np.repeat(np.expand_dims(np.linspace(0, widthE - 1, widthE), 0), heightE, 0) / heightE * np.pi - 3 * np.pi / 2  #-3*pi/2:1/2;-4pi/2:1/3
  theta = np.repeat((np.expand_dims(np.linspace(0, heightE - 1, heightE), 0).T), widthE, 1) / heightE * np.pi - np.pi / 2
  print(phi.shape)
  print(phi)
  erpLeft = np.ndarray([heightE, widthE, 3])
  erpRight = np.ndarray([heightE, widthE, 3])
  x = np.cos(theta) * np.cos(phi)
  y = np.cos(theta) * np.sin(phi)
  z = np.sin(theta)
  phi2 = np.arcsin(x)
  theta2 = np.arctan2(z, y)
  print(np.max(phi2), np.min(phi2))
  print(np.max(theta2), np.min(theta2))
  print(phi2)
  u = (((phi2 + np.pi / 2) / np.pi * heightE).astype(np.int) + widthC) % widthC
  print(np.max(u), np.min(u))
  v = (((-theta2) / np.pi * heightE).astype(np.int) + heightC) % heightC
  u[u > widthC - 1] = widthC - 1
  v[v > heightC - 1] = heightC - 1
  for i in range(heightE):
    for j in range(widthE):
      ii = u[i, j]
      jj = v[i, j]
      erpLeft[i, j, :] = caleft[jj, ii, :]
      erpRight[i, j, :] = caright[jj, ii, :]
  print(erpLeft.shape)
  print(erpLeft.dtype)
  print(np.max(erpLeft), np.min(erpLeft))
  cv2.imwrite('erpLeft.png', erpLeft)
  cv2.imwrite('erpRight.png', erpRight)
  return erpLeft, erpRight


if __name__ == "__main__":
  root = "../../tmp/"
  # e2ca = ERP2CA(256, 512, 512, 256, False)
  # imgName = "0_0b217f59904d4bdf85d35da2cab963471_color_0_Left_Down_0.0.png"
  # left = cv2.imread(os.path.join(root, imgName), cv2.IMREAD_ANYCOLOR)
  # left = left.transpose((2, 0, 1)).astype(np.float32)
  # left = torch.from_numpy(left).unsqueeze_(0)
  # l = e2ca.trans(left, '0')
  # lsave = l.squeeze(0).numpy().transpose((1, 2, 0))
  # cv2.imwrite(os.path.join(root, 'e2ca_' + imgName), lsave)
  ca2e1 = CA2ERP(64, 128, 128, 64, False, False)
  ca2e2 = CA2ERP(64, 128, 128, 64, False, True)
  e2ca = ERP2CA(512, 1024, 1024, 512, False)
  imgName = "ca2e_002505_34_4.png"
  left = cv2.imread(os.path.join(root, imgName))
  left = left.transpose((2, 0, 1)).astype(np.float32)
  print(left.shape)

  roll_idx = int(1024 / 360 * -135)
  left = np.roll(left, roll_idx, 2)
  left[:, :, 0:256] = 0
  left[:, :, 768:] = 0
  leftsave = left.transpose((1, 2, 0)).astype(np.uint8)
  print(leftsave.shape)

  cv2.imwrite(os.path.join(root, 'masked_' + imgName), leftsave)
  left = torch.from_numpy(left).unsqueeze_(0)
  l = e2ca.trans(left, '0')
  lsave = l.squeeze(0).numpy().transpose((1, 2, 0))
  print(lsave.shape)
  cv2.imwrite(os.path.join(root, 'e2ca_' + imgName), lsave)
  #print(torch.sum(torch.abs(ca2e1.gridDict['0'] - ca2e2.gridDict['0'])))
  '''
  imgName = "008258"
  # cam_pairs = ['12', '13', '14', '23', '24', '34']
  # angles = ['0', '90', '45', '135', '90', '0']
  cam_pairs = ['12', '34']
  angles = ['0', '0']
  # left = cv2.imread(os.path.join(root, imgName), cv2.IMREAD_ANYCOLOR)
  # for i in range(16, 0, -1):
  #   l0 = ErpRoll(left, -i * np.pi / 8)
  #   cv2.imwrite(os.path.join(root, 'trans_' + str(i) + '_' + imgName), l0)
  e2ca = ERP2CA(512, 1024, 1024, 512)
  ca2e = CA2ERP(512, 1024, 1024, 512)
  for i in range(len(cam_pairs)):
    cp = cam_pairs[i]
    a = '0'
    left = cv2.imread(os.path.join(root, imgName + '_' + cp + '_rgb' + cp[0] + '.png'), cv2.IMREAD_ANYCOLOR)
    right = cv2.imread(os.path.join(root, imgName + '_' + cp + '_rgb' + cp[1] + '.png'), cv2.IMREAD_ANYCOLOR)
    left = left.transpose((2, 0, 1)).astype(np.float32)
    left = torch.from_numpy(left).unsqueeze_(0)
    right = right.transpose((2, 0, 1)).astype(np.float32)
    right = torch.from_numpy(right).unsqueeze_(0)
    l, r = ca2e.transPairs(left, right, a)
    lsave = l.squeeze(0).numpy().transpose((1, 2, 0))
    rsave = r.squeeze(0).numpy().transpose((1, 2, 0))
    cv2.imwrite(os.path.join(root, 'ca2e_' + imgName + '_' + cp + '_' + cp[0] + '.png'), lsave)
    cv2.imwrite(os.path.join(root, 'ca2e_' + imgName + '_' + cp + '_' + cp[1] + '.png'), rsave)
    le, re = e2ca.transPairs(l, r, a)
    lsave = le.squeeze(0).numpy().transpose((1, 2, 0))
    rsave = re.squeeze(0).numpy().transpose((1, 2, 0))
    cv2.imwrite(os.path.join(root, 'e2ca_' + imgName + '_' + cp + '_' + cp[0] + '.png'), lsave)
    cv2.imwrite(os.path.join(root, 'e2ca_' + imgName + '_' + cp + '_' + cp[1] + '.png'), rsave)
  '''
