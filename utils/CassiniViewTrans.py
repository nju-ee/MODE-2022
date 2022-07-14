import os
import sys
import cv2
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from multiprocessing import Pool, sharedctypes
from numba import jit

# from preprocess import *
defaultConfigs = {
    'height':
    1024,
    'width':
    512,
    'cam_pair_num':
    6,
    'baselines': [1,
                  1,
                  math.sqrt(2),
                  math.sqrt(2),
                  1,
                  1],
    'trans_rotate': [
        #y,z,x,pitch,yaw,roll
        #12
        [0,
         0,
         0,
         0,
         0,
         0],
        #13
        [0,
         0,
         0,
         0.5 * math.pi,
         0,
         0],
        #14
        [0,
         0,
         0,
         0.25 * math.pi,
         0,
         0],
        #23
        [0,
         -math.sqrt(2) / 2,
         -math.sqrt(2) / 2,
         0.75 * math.pi,
         0,
         0],
        #24
        [0,
         -1,
         0,
         0.5 * math.pi,
         0,
         0],
        #34
        [0,
         1,
         0,
         0,
         0,
         0]
    ]
}

depthGtTransConfigs = {
    'height':
    1024,
    'width':
    512,
    'cam_pair_num':
    6,
    'baselines': [1,
                  1,
                  math.sqrt(2),
                  math.sqrt(2),
                  1,
                  1],
    'trans_rotate': [
        #y,z,x,pitch,yaw,roll
        #12
        [0,
         0,
         0,
         0,
         0,
         0],
        #13
        [0,
         0,
         0,
         -0.5 * math.pi,
         0,
         0],
        #14
        [0,
         0,
         0,
         -0.25 * math.pi,
         0,
         0],
        #23
        [0,
         0,
         -1,
         -0.75 * math.pi,
         0,
         0],
        #24
        [0,
         0,
         -1,
         -0.5 * math.pi,
         0,
         0],
        #34
        [0,
         -1,
         0,
         0,
         0,
         0]
    ]
}

omniFisheyeConfigs = {
    'height':
    640,
    'width':
    320,
    'cam_pair_num':
    6,
    'baselines': [0.6 * np.sqrt(2),
                  0.6 * np.sqrt(2),
                  1.2,
                  1.2,
                  0.6 * np.sqrt(2),
                  0.6 * np.sqrt(2)],
    'trans_rotate': [
        #y,z,x,pitch,yaw,roll
        #12
        [0,
         -0.6 / np.sqrt(2),
         -0.6 / np.sqrt(2),
         0,
         0,
         0],
        #13
        [0,
         0.6 / np.sqrt(2),
         -0.6 / np.sqrt(2),
         0.5 * np.pi,
         0,
         0],
        #14
        [0,
         0,
         -0.6,
         0.25 * np.pi,
         0,
         0],
        #23
        [0,
         0,
         -0.6,
         0.75 * np.pi,
         0,
         0],
        #24
        [0,
         -0.6 / np.sqrt(2),
         -0.6 / np.sqrt(2),
         0.5 * np.pi,
         0,
         0],
        #34
        [0,
         0.6 / np.sqrt(2),
         -0.6 / np.sqrt(2),
         0,
         0,
         0]
    ]
}

configs = defaultConfigs


class CassiniViewDepthTransformer():
  def __init__(self, configs=defaultConfigs):
    self.cam_pair_num = configs['cam_pair_num']
    self.height, self.width = configs['height'], configs['width']
    self.baselines = configs['baselines']
    self.trans_rotate = configs['trans_rotate']
    assert self.cam_pair_num == len(self.baselines)
    assert self.cam_pair_num == len(self.trans_rotate)
    theta_1_start = np.pi - (np.pi / self.height)
    theta_1_end = -np.pi
    theta_1_step = 2 * np.pi / self.height
    theta_1_range = np.arange(theta_1_start, theta_1_end, -theta_1_step)
    self.theta_1_map = np.array([theta_1_range for i in range(self.width)]).astype(np.float32).T

    phi_1_start = 0.5 * np.pi - (0.5 * np.pi / self.width)
    phi_1_end = -0.5 * np.pi
    phi_1_step = np.pi / self.width
    phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
    self.phi_1_map = np.array([phi_1_range for j in range(self.height)]).astype(np.float32)
    self.t = []
    self.R = []
    for i in range(self.cam_pair_num):
      y0, z0, x0, pitch, yaw, roll = self.trans_rotate[i]
      self.t.append(torch.from_numpy(np.array([[x0], [y0], [z0]]).astype(np.float32)))
      Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
      Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
      Ry = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
      Ra = np.dot(np.dot(Rx, Rz), Ry)
      self.R.append(torch.from_numpy(np.linalg.inv(Ra).astype(np.float32)))

  def dispToDepthTrans(self, disp, camPair, maxDisp=208, maxDepth=1000, device='cuda'):
    n, c, h, w = disp.shape
    assert camPair >= 0 and camPair < self.cam_pair_num
    t0 = self.t[camPair].to(device)
    R0 = self.R[camPair].to(device)
    bl0 = self.baselines[camPair]
    validMask = (disp > 0) & (~torch.isnan(disp)) & (~torch.isinf(disp)) & (disp <= maxDisp)
    depth = torch.zeros_like(disp)
    dispPhi = torch.zeros_like(disp)
    dispPhi[validMask] = disp[validMask] * np.pi / self.width
    phi_l_map = torch.from_numpy(self.phi_1_map).unsqueeze_(0).unsqueeze_(0).to(device).repeat(n, 1, 1, 1)
    theta_l_map = torch.from_numpy(self.theta_1_map).unsqueeze_(0).unsqueeze_(0).to(device).repeat(n, 1, 1, 1)
    phi_r_map = phi_l_map + dispPhi
    depth[validMask] = bl0 * torch.sin(torch.ones_like(phi_r_map[validMask]) * np.pi / 2 - phi_r_map[validMask]) / torch.sin(phi_r_map[validMask] - phi_l_map[validMask])
    # x1 = depth * torch.sin(phi_l_map)
    # y1 = depth * torch.cos(phi_l_map) * torch.sin(theta_l_map)
    # z1 = depth * torch.cos(phi_l_map) * torch.cos(theta_l_map)
    # X1 = torch.cat([x1, y1, z1], 1).view(n, 3, -1)
    X1 = torch.cat([depth * torch.sin(phi_l_map), depth * torch.cos(phi_l_map) * torch.sin(theta_l_map), depth * torch.cos(phi_l_map) * torch.cos(theta_l_map)], 1).view(n, 3, -1)
    X2 = torch.matmul(R0, (X1 - t0.repeat(n, 1, 1))).view(n, 3, h, w)
    depth2 = torch.sqrt(X2[:, 0, :, :]**2 + X2[:, 1, :, :]**2 + X2[:, 2, :, :]**2).unsqueeze_(1)
    validMask2 = (depth2 > 0) & (~torch.isnan(depth2)) & (~torch.isinf(depth2)) & (depth2 <= maxDepth)
    theta_l_map2 = torch.zeros_like(depth2)
    phi_l_map2 = torch.zeros_like(depth2)
    theta_l_map2 = torch.atan2(X2[:, 1:2, :, :], X2[:, 2:, :, :])
    phi_l_map2[validMask2] = torch.asin(torch.clamp(X2[:, 0:1, :, :][validMask2] / depth2[validMask2], -1, 1))
    theta_l_map2 = (theta_l_map2 - torch.min(theta_l_map2)) / (torch.max(theta_l_map2) - torch.min(theta_l_map2)) * 2 - 1
    phi_l_map2 = (phi_l_map2 - torch.min(phi_l_map2)) / (torch.max(phi_l_map2) - torch.min(phi_l_map2)) * 2 - 1
    grid = torch.cat([-phi_l_map2, -theta_l_map2], 1).permute(0, 2, 3, 1)
    depthTrans = F.grid_sample(depth2, grid, mode='bilinear', align_corners=True, padding_mode='border')
    depthTrans[~validMask2] = 0  # set the depth values of invalid points to 0.
    return depthTrans

  def dispBasedRGBTrans(self, disp, camPair, RGB):
    n, c, h, w = disp.shape
    assert camPair >= 0 and camPair < self.cam_pair_num
    t0 = self.t[camPair]
    R0 = self.R[camPair]
    bl0 = self.baselines[camPair]
    validMask = (disp > 0) & (~torch.isnan(disp)) & (~torch.isinf(disp))
    depth = torch.zeros_like(disp)
    dispPhi = torch.zeros_like(disp)
    dispPhi[validMask] = disp[validMask] * np.pi / self.width
    phi_l_map = torch.from_numpy(self.phi_1_map).unsqueeze_(0).unsqueeze_(0)
    theta_l_map = torch.from_numpy(self.theta_1_map).unsqueeze_(0).unsqueeze_(0)
    phi_r_map = phi_l_map + dispPhi
    depth[validMask] = bl0 * torch.sin(np.pi / 2 - phi_r_map[validMask]) / np.sin(phi_r_map[validMask] - phi_l_map[validMask])
    x1 = depth * torch.sin(phi_l_map)
    y1 = depth * torch.cos(phi_l_map) * np.sin(theta_l_map)
    z1 = depth * torch.cos(phi_l_map) * np.cos(theta_l_map)
    X1 = torch.cat([x1, y1, z1], 1).view(n, 3, -1)
    X2 = torch.matmul(R0, (X1 - t0.repeat(n, 1, 1))).view(n, 3, h, w)
    depth2 = torch.sqrt(X2[:, 0, :, :]**2 + X2[:, 1, :, :]**2 + X2[:, 2, :, :]**2).unsqueeze_(1)
    validMask2 = (depth2 > 0) & (~torch.isnan(depth2)) & (~torch.isinf(depth2))
    theta_l_map2 = torch.zeros_like(depth2)
    phi_l_map2 = torch.zeros_like(depth2)
    theta_l_map2 = torch.atan2(X2[:, 1:2, :, :], X2[:, 2:, :, :])
    phi_l_map2[validMask2] = torch.asin(torch.clamp(X2[:, 0:1, :, :][validMask2] / depth2[validMask2], -1, 1))
    theta_l_map2 = (theta_l_map2 - torch.min(theta_l_map2)) / (torch.max(theta_l_map2) - torch.min(theta_l_map2)) * 2 - 1
    phi_l_map2 = (phi_l_map2 - torch.min(phi_l_map2)) / (torch.max(phi_l_map2) - torch.min(phi_l_map2)) * 2 - 1
    grid = torch.cat([-phi_l_map2, -theta_l_map2], 1).permute(0, 2, 3, 1)
    RGBOut = F.grid_sample(RGB, grid, mode='bilinear', align_corners=True, padding_mode='border')
    validMask2 = validMask2.repeat(1, 3, 1, 1)
    RGBOut[~validMask2] = 0.0
    return RGBOut


def Cassini2Cassini(cassini_1, pitch, yaw, roll):
  Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])

  Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])

  Ry = np.array([[math.cos(pitch), 0, -math.sin(pitch)], [0, 1, 0], [math.sin(pitch), 0, math.cos(pitch)]])

  R = np.dot(np.dot(Rx, Rz), Ry)
  R_I = np.linalg.inv(R)

  output_h = cassini_1.shape[0]
  output_w = cassini_1.shape[1]

  theta_2_start = math.pi - (math.pi / output_h)
  theta_2_end = -math.pi
  theta_2_step = 2 * math.pi / output_h
  theta_2_range = np.arange(theta_2_start, theta_2_end, -theta_2_step)
  theta_2_map = np.array([theta_2_range for i in range(output_w)]).astype(np.float32).T

  phi_2_start = 0.5 * math.pi - (0.5 * math.pi / output_w)
  phi_2_end = -0.5 * math.pi
  phi_2_step = math.pi / output_w
  phi_2_range = np.arange(phi_2_start, phi_2_end, -phi_2_step)
  phi_2_map = np.array([phi_2_range for j in range(output_h)]).astype(np.float32)

  x_2 = np.sin(phi_2_map)
  y_2 = np.cos(phi_2_map) * np.sin(theta_2_map)
  z_2 = np.cos(phi_2_map) * np.cos(theta_2_map)
  X_2 = np.expand_dims(np.dstack((x_2, y_2, z_2)), axis=-1)
  X_1 = np.matmul(R_I, X_2)
  theta_1_map = np.arctan2(X_1[:, :, 1, 0], X_1[:, :, 2, 0])
  phi_1_map = np.arcsin(np.clip(X_1[:, :, 0, 0], -1, 1))

  source_image = Variable(torch.FloatTensor(cassini_1).unsqueeze(0).transpose(1, 3).transpose(2, 3))
  grid_x = Variable(torch.FloatTensor(np.clip(-phi_1_map / (0.5 * math.pi), -1, 1)).unsqueeze(-1))
  grid_y = Variable(torch.FloatTensor(np.clip(-theta_1_map / math.pi, -1, 1)).unsqueeze(-1))

  grid = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0)
  sampled_image = torch.nn.functional.grid_sample(source_image, grid, mode='bilinear', align_corners=True, padding_mode='border')  # 1, ch, self.output_h, self.output_w

  cassini_2 = sampled_image.transpose(1, 3).transpose(1, 2).data.cpu().numpy()[0].astype(np.float32)
  return cassini_2


def View_trans_jit(view_1, y0, z0, x0, pitch, yaw, roll):
  Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])

  Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])

  Ry = np.array([[math.cos(pitch), 0, -math.sin(pitch)], [0, 1, 0], [math.sin(pitch), 0, math.cos(pitch)]])

  R = np.dot(np.dot(Rx, Rz), Ry)

  t = np.array([[x0], [y0], [z0]])

  output_h = view_1.shape[0]
  output_w = view_1.shape[1]

  theta_1_start = math.pi - (math.pi / output_h)
  theta_1_end = -math.pi
  theta_1_step = 2 * math.pi / output_h
  theta_1_range = np.arange(theta_1_start, theta_1_end, -theta_1_step)
  theta_1_map = np.array([theta_1_range for i in range(output_w)]).astype(np.float32).T

  phi_1_start = 0.5 * math.pi - (0.5 * math.pi / output_w)
  phi_1_end = -0.5 * math.pi
  phi_1_step = math.pi / output_w
  phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
  phi_1_map = np.array([phi_1_range for j in range(output_h)]).astype(np.float32)

  r_1 = view_1

  x_1 = r_1 * np.sin(phi_1_map)
  y_1 = r_1 * np.cos(phi_1_map) * np.sin(theta_1_map)
  z_1 = r_1 * np.cos(phi_1_map) * np.cos(theta_1_map)
  X_1 = np.expand_dims(np.dstack((x_1, y_1, z_1)), axis=-1)

  X_2 = np.matmul(R, X_1 - t)

  r_2 = np.sqrt(np.square(X_2[:, :, 0, 0]) + np.square(X_2[:, :, 1, 0]) + np.square(X_2[:, :, 2, 0]))
  theta_2_map = np.arctan2(X_2[:, :, 1, 0], X_2[:, :, 2, 0])
  phi_2_map = np.arcsin(np.clip(X_2[:, :, 0, 0] / r_2, -1, 1))

  view_2 = np.ones((output_h, output_w)).astype(np.float32) * 100000

  I_2 = np.clip(np.rint(output_h / 2 - output_h * theta_2_map / (2 * math.pi)), 0, output_h - 1).astype(np.int16)
  J_2 = np.clip(np.rint(output_w / 2 - output_w * phi_2_map / math.pi), 0, output_w - 1).astype(np.int16)

  view_2 = iter_pixels(output_h, output_w, r_2, view_2, I_2, J_2)

  view_2[view_2 == 100000] = 0
  view_2 = view_2.astype(np.float32)

  return view_2


@jit(nopython=True)
def iter_pixels(output_h, output_w, r_2, view_2, I_2, J_2):
  for i in range(output_h):
    for j in range(output_w):
      flag = r_2[i, j] < view_2[I_2[i, j], J_2[i, j]]
      view_2[I_2[i, j], J_2[i, j]] = flag * r_2[i, j] + (1 - flag) * view_2[I_2[i, j], J_2[i, j]]
  return view_2


def View_trans(view_1, y0, z0, x0, pitch, yaw, roll):
  Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])

  Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])

  Ry = np.array([[math.cos(pitch), 0, -math.sin(pitch)], [0, 1, 0], [math.sin(pitch), 0, math.cos(pitch)]])

  R = np.dot(np.dot(Rx, Rz), Ry)

  t = np.array([[x0], [y0], [z0]])

  output_h = view_1.shape[0]
  output_w = view_1.shape[1]

  theta_1_start = math.pi - (math.pi / output_h)
  theta_1_end = -math.pi
  theta_1_step = 2 * math.pi / output_h
  theta_1_range = np.arange(theta_1_start, theta_1_end, -theta_1_step)
  theta_1_map = np.array([theta_1_range for i in range(output_w)]).astype(np.float32).T

  phi_1_start = 0.5 * math.pi - (0.5 * math.pi / output_w)
  phi_1_end = -0.5 * math.pi
  phi_1_step = math.pi / output_w
  phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
  phi_1_map = np.array([phi_1_range for j in range(output_h)]).astype(np.float32)

  r_1 = view_1

  x_1 = r_1 * np.sin(phi_1_map)
  y_1 = r_1 * np.cos(phi_1_map) * np.sin(theta_1_map)
  z_1 = r_1 * np.cos(phi_1_map) * np.cos(theta_1_map)
  X_1 = np.expand_dims(np.dstack((x_1, y_1, z_1)), axis=-1)

  X_2 = np.matmul(R, X_1 - t)

  r_2 = np.sqrt(np.square(X_2[:, :, 0, 0]) + np.square(X_2[:, :, 1, 0]) + np.square(X_2[:, :, 2, 0]))
  theta_2_map = np.arctan2(X_2[:, :, 1, 0], X_2[:, :, 2, 0])
  phi_2_map = np.arcsin(np.clip(X_2[:, :, 0, 0] / r_2, -1, 1))

  view_2 = np.ones((output_h, output_w)).astype(np.float32) * np.inf

  for i in range(output_h):
    for j in range(output_w):
      i_2 = int(np.clip(np.rint(output_h / 2 - output_h * theta_2_map[i, j] / (2 * math.pi)), 0, output_h - 1))
      j_2 = int(np.clip(np.rint(output_w / 2 - output_w * phi_2_map[i, j] / math.pi), 0, output_w - 1))
      if (r_2[i, j] < view_2[i_2, j_2]):
        view_2[i_2, j_2] = r_2[i, j]

  view_2[view_2 == np.inf] = 0

  return view_2


def View_trans_base(view_1, y0, z0, x0, pitch, yaw, roll):
  Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])

  Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])

  Ry = np.array([[math.cos(pitch), 0, -math.sin(pitch)], [0, 1, 0], [math.sin(pitch), 0, math.cos(pitch)]])

  R = np.dot(np.dot(Rx, Rz), Ry)

  t = np.array([[x0], [y0], [z0]])

  output_h = view_1.shape[0]
  output_w = view_1.shape[1]

  theta_1_start = math.pi - (math.pi / output_h)
  theta_1_end = -math.pi
  theta_1_step = 2 * math.pi / output_h
  theta_1_range = np.arange(theta_1_start, theta_1_end, -theta_1_step)
  theta_1_map = np.array([theta_1_range for i in range(output_w)]).astype(np.float32).T

  phi_1_start = 0.5 * math.pi - (0.5 * math.pi / output_w)
  phi_1_end = -0.5 * math.pi
  phi_1_step = math.pi / output_w
  phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
  phi_1_map = np.array([phi_1_range for j in range(output_h)]).astype(np.float32)

  r_1 = view_1

  x_1 = r_1 * np.sin(phi_1_map)
  y_1 = r_1 * np.cos(phi_1_map) * np.sin(theta_1_map)
  z_1 = r_1 * np.cos(phi_1_map) * np.cos(theta_1_map)
  X_1 = np.expand_dims(np.dstack((x_1, y_1, z_1)), axis=-1)

  X_2 = np.matmul(R, X_1 - t)

  r_2 = np.sqrt(np.square(X_2[:, :, 0, 0]) + np.square(X_2[:, :, 1, 0]) + np.square(X_2[:, :, 2, 0]))
  theta_2_map = np.arctan2(X_2[:, :, 1, 0], X_2[:, :, 2, 0])
  phi_2_map = np.arcsin(np.clip(X_2[:, :, 0, 0] / r_2, -1, 1))

  I_2 = np.clip(np.rint(output_h / 2 - output_h * theta_2_map / (2 * math.pi)), 0, output_h - 1).astype(np.int16)
  J_2 = np.clip(np.rint(output_w / 2 - output_w * phi_2_map / math.pi), 0, output_w - 1).astype(np.int16)

  return r_2, I_2, J_2


def __View_trans_parallel_a_colum(j):
  for i in range(output_h_share.value):
    flag = r_2[i][j] < view_2[I_2[i][j]][J_2[i][j]]
    view_2[I_2[i][j]][J_2[i][j]] = flag * r_2[i][j] + (1 - flag) * view_2[I_2[i][j]][J_2[i][j]]


def View_trans_parallel(view_1, y0, z0, x0, pitch, yaw, roll):
  global output_h_share, r_2, view_2, I_2, J_2
  output_h = view_1.shape[0]
  output_w = view_1.shape[1]

  output_h_share = sharedctypes.RawValue('i', output_h)

  view_2_tmp = np.ones((output_h, output_w)).astype(np.float32) * 100000
  view_2_tmp = np.ctypeslib.as_ctypes(view_2_tmp)
  view_2 = sharedctypes.RawArray(view_2_tmp._type_, view_2_tmp)

  r_2, I_2, J_2 = View_trans_base(view_1, y0, z0, x0, pitch, yaw, roll)

  r_2 = np.ctypeslib.as_ctypes(r_2)
  r_2 = sharedctypes.RawArray(r_2._type_, r_2)

  I_2 = np.ctypeslib.as_ctypes(I_2)
  I_2 = sharedctypes.RawArray(I_2._type_, I_2)

  J_2 = np.ctypeslib.as_ctypes(J_2)
  J_2 = sharedctypes.RawArray(J_2._type_, J_2)

  pool = Pool(1)
  pool.map(__View_trans_parallel_a_colum, range(output_w))
  pool.close()
  pool.join()

  view_2 = np.ctypeslib.as_array(view_2)
  view_2 = view_2.reshape(output_h, output_w)
  view_2[view_2 == 100000] = 0
  view_2 = view_2.astype(np.float32)
  return view_2.copy()


def disp2depth(disp, cam_pair, max_depth=1000, parallel=True, configs=defaultConfigs, keep_view=False):
  if parallel:
    transFunc = View_trans_jit
  else:
    transFunc = View_trans
    # B = np.array([1, 1, math.sqrt(2), math.sqrt(2), 1, 1]).astype(np.float32)
  B = np.array(configs['baselines']).astype(np.float32)

  output_h = disp.shape[0]
  output_w = disp.shape[1]

  phi_l_start = 0.5 * math.pi - (0.5 * math.pi / output_w)
  phi_l_end = -0.5 * math.pi
  phi_l_step = math.pi / output_w
  phi_l_range = np.arange(phi_l_start, phi_l_end, -phi_l_step)
  phi_l_map = np.array([phi_l_range for j in range(output_h)]).astype(np.float32)

  mask_disp_is_0 = disp == 0
  disp_not_0 = np.ma.array(disp, mask=mask_disp_is_0)

  phi_r_map = disp_not_0 * math.pi / output_w + phi_l_map

  # sin theory
  depth_l = B[cam_pair] * np.sin(math.pi / 2 - phi_r_map) / np.sin(phi_r_map - phi_l_map)
  depth_l = depth_l.filled(max_depth)
  depth_l[depth_l > max_depth] = max_depth
  depth_l[depth_l < 0] = 0

  if keep_view:
    return depth_l

  y0, z0, x0, pitch, yaw, roll = configs['trans_rotate'][cam_pair]

  if cam_pair == 0:
    return depth_l
  elif cam_pair == 1:
    cassini_1 = np.expand_dims(depth_l, axis=-1)
    cassini_2 = Cassini2Cassini(cassini_1, pitch, yaw, roll)
    return cassini_2[:, :, 0]
  elif cam_pair == 2:
    cassini_1 = np.expand_dims(depth_l, axis=-1)
    cassini_2 = Cassini2Cassini(cassini_1, pitch, yaw, roll)
    return cassini_2[:, :, 0]
  elif cam_pair == 3:
    view_2 = transFunc(depth_l, y0, z0, x0, pitch, yaw, roll)
    return view_2
  elif cam_pair == 4:
    view_2 = transFunc(depth_l, y0, z0, x0, pitch, yaw, roll)
    return view_2
  elif cam_pair == 5:
    view_2 = transFunc(depth_l, y0, z0, x0, pitch, yaw, roll)
    return view_2
  else:
    return None


def batchToDepth(batchDisp, camPair, maxDepth=1000, device='cuda'):
  batch, c, height, width = batchDisp.shape
  phi_1_start = 0.5 * math.pi - (0.5 * math.pi / width)
  phi_1_end = -0.5 * math.pi
  phi_1_step = math.pi / width
  phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
  phi_1_map = np.array([phi_1_range for j in range(height)]).astype(np.float32)
  baseline = defaultConfigs['baselines'][camPair]
  validMask = batchDisp > 0
  depth = torch.ones_like(batchDisp) * maxDepth
  dispPhi = batchDisp * np.pi / width
  phi_l_map = torch.from_numpy(phi_1_map).unsqueeze_(0).unsqueeze_(0).to(device).repeat(batch, 1, 1, 1)
  #theta_l_map = torch.from_numpy(theta_1_map).unsqueeze_(0).unsqueeze_(0).to(device).repeat(n, 1, 1, 1)
  phi_r_map = phi_l_map + dispPhi
  depth[validMask] = baseline * torch.sin(torch.ones_like(phi_r_map[validMask]) * np.pi / 2 - phi_r_map[validMask]) / torch.sin(phi_r_map[validMask] - phi_l_map[validMask])
  depth[depth < 0] = 0
  depth[~validMask] = maxDepth
  depth[depth > maxDepth] = maxDepth
  return depth


def batchCassini2Cassini(batchDepth, camPair, maxDepth=1000, device='cuda'):
  batch, c, height, width = batchDepth.shape
  y0, z0, x0, pitch, yaw, roll = defaultConfigs['trans_rotate'][camPair]
  Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
  Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
  Ry = np.array([[math.cos(pitch), 0, -math.sin(pitch)], [0, 1, 0], [math.sin(pitch), 0, math.cos(pitch)]])
  R = np.dot(np.dot(Rx, Rz), Ry)
  R_I = np.linalg.inv(R).astype(np.float32)
  R_I = torch.from_numpy(R_I).to(device)
  theta_1_start = np.pi - (np.pi / height)
  theta_1_end = -np.pi
  theta_1_step = 2 * np.pi / height
  theta_1_range = np.arange(theta_1_start, theta_1_end, -theta_1_step)
  theta_1_map = np.array([theta_1_range for i in range(width)]).astype(np.float32).T

  phi_1_start = 0.5 * np.pi - (0.5 * np.pi / width)
  phi_1_end = -0.5 * np.pi
  phi_1_step = np.pi / width
  phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
  phi_1_map = np.array([phi_1_range for j in range(height)]).astype(np.float32)
  phi_1_map = torch.from_numpy(phi_1_map).unsqueeze_(0).unsqueeze_(0).to(device).repeat(batch, 1, 1, 1)
  theta_1_map = torch.from_numpy(theta_1_map).unsqueeze_(0).unsqueeze_(0).to(device).repeat(batch, 1, 1, 1)
  # x_2 = np.sin(phi_1_map)
  # y_2 = np.cos(phi_1_map) * np.sin(theta_1_map)
  # z_2 = np.cos(phi_1_map) * np.cos(theta_1_map)
  # X1 = np.expand_dims(np.dstack((x_2, y_2, z_2)), axis=-1)
  # X2 = np.matmul(R_I, X1)

  # theta_2_map = np.arctan2(X2[:, :, 1, 0], X2[:, :, 2, 0])
  # phi_2_map = np.arcsin(np.clip(X2[:, :, 0, 0], -1, 1))

  # grid_x = Variable(torch.FloatTensor(np.clip(-phi_1_map / (0.5 * math.pi), -1, 1)).unsqueeze(-1))
  # grid_y = Variable(torch.FloatTensor(np.clip(-theta_1_map / math.pi, -1, 1)).unsqueeze(-1))

  # grid = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1).to(device)

  X1 = torch.cat([torch.sin(phi_1_map), torch.cos(phi_1_map) * torch.sin(theta_1_map), torch.cos(phi_1_map) * torch.cos(theta_1_map)], 1).view(batch, 3, -1)
  X2 = torch.matmul(R_I, X1).view(batch, 3, height, width)
  # depth2 = torch.sqrt(torch.pow(X2[:, 0, :, :], 2) + torch.pow(X2[:, 1, :, :], 2) + torch.pow(X2[:, 2, :, :], 2)).unsqueeze_(1)
  # validMask2 = (depth2 > 0) & (~torch.isnan(depth2)) & (~torch.isinf(depth2)) & (depth2 <= maxDepth)
  # validMask2 = (~torch.isnan(depth2)) & (~torch.isinf(depth2))
  theta_l_map2 = torch.atan2(X2[:, 1:2, :, :], X2[:, 2:, :, :])
  phi_l_map2 = torch.asin(torch.clamp(X2[:, 0:1, :, :], -1, 1))
  # theta_l_map2 = (theta_l_map2 - torch.min(theta_l_map2)) / (torch.max(theta_l_map2) - torch.min(theta_l_map2)) * 2 - 1
  # phi_l_map2 = (phi_l_map2 - torch.min(phi_l_map2)) / (torch.max(phi_l_map2) - torch.min(phi_l_map2)) * 2 - 1
  theta_l_map2 = torch.clamp(theta_l_map2 / (1 * math.pi), -1, 1)
  phi_l_map2 = torch.clamp(phi_l_map2 / (0.5 * math.pi), -1, 1)
  grid = torch.cat([-phi_l_map2, -theta_l_map2], 1).permute(0, 2, 3, 1)
  depthTrans = F.grid_sample(batchDepth, grid, mode='bilinear', align_corners=True, padding_mode='border')
  #depthTrans[~validMask2] = maxDepth
  return depthTrans


def batchViewTrans(batchDepth, camPair, maxDepth=1000, parallel=True, device='cuda'):
  batch, c, height, width = batchDepth.shape
  y0, z0, x0, pitch, yaw, roll = defaultConfigs['trans_rotate'][camPair]
  if parallel:
    transFunc = View_trans_parallel
  else:
    transFunc = View_trans
  transList = []
  for i in range(batch):
    dep = batchDepth[i, ::].squeeze(0).squeeze(0).cpu().numpy()
    depT = transFunc(dep, y0, z0, x0, pitch, yaw, roll)
    transList.append(torch.from_numpy(depT).unsqueeze_(0).unsqueeze_(0))
  depthTrans = torch.cat(transList, dim=0)
  if device == 'cuda':
    depthTrans = depthTrans.cuda()
  return depthTrans


def batchDisp2Depth(batchDisp, cam_pair, max_depth=1000, parallel=True, device='cuda'):
  batchDepth = batchToDepth(batchDisp, cam_pair, max_depth, device)
  if (cam_pair == 0):
    batchDepthTrans = batchDepth
  elif (cam_pair <= 2):
    batchDepthTrans = batchCassini2Cassini(batchDepth, cam_pair, max_depth, device)
  else:
    batchDepthTrans = batchViewTrans(batchDepth, cam_pair, max_depth, parallel, device)
  return batchDepthTrans


# view trans. convert disparity maps on other view to depth maps on view 1
def batchViewTransOri(batchDisp, camPair, max_depth=1000, cpuParallel=True, cuda=True, configs=defaultConfigs):
  n, c, h, w = batchDisp.shape
  tmp = []
  for j in range(n):
    disp_j = batchDisp[j, ::].squeeze(0).squeeze(0).cpu().numpy()
    depth_j = disp2depth(disp_j, camPair, max_depth, cpuParallel, configs)
    tmp.append(torch.from_numpy(depth_j).unsqueeze(0).unsqueeze(0))
  depthOut = torch.cat(tmp, dim=0)
  if cuda:
    depthOut = depthOut.cuda()
  return depthOut


def depthGtViewTrans(depthGT, cam_pair, configs=defaultConfigs):
  y0, z0, x0, pitch, yaw, roll = configs['trans_rotate'][cam_pair]
  if cam_pair == 0:
    return depthGT
  elif cam_pair == 1:
    cassini_1 = np.expand_dims(depthGT, axis=-1)
    cassini_2 = Cassini2Cassini(cassini_1, pitch, yaw, roll)
    return cassini_2[:, :, 0]
  elif cam_pair == 2:
    cassini_1 = np.expand_dims(depthGT, axis=-1)
    cassini_2 = Cassini2Cassini(cassini_1, pitch, yaw, roll)
    return cassini_2[:, :, 0]
  elif cam_pair == 3:
    view_2 = View_trans_jit(depthGT, y0, z0, x0, pitch, yaw, roll)
    return view_2
  elif cam_pair == 4:
    view_2 = View_trans_jit(depthGT, y0, z0, x0, pitch, yaw, roll)
    return view_2
  elif cam_pair == 5:
    view_2 = View_trans_jit(depthGT, y0, z0, x0, pitch, yaw, roll)
    return view_2
  else:
    return None


if __name__ == '__main__':
  from ERPandCassini import CA2ERP
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  CVDT = CassiniViewDepthTransformer(defaultConfigs)
  root = '/home/liming/projects/datasets/Deep360/depth_on_1/ep1_500frames/training'
  rootsave = '../../tmp'
  name = '002530'
  viewNames = ['002530_12_rgb1.png', '002530_13_rgb1.png', '002530_14_rgb1.png', '002530_23_rgb2.png', '002530_24_rgb2.png', '002530_34_rgb3.png']
  dispNames = ['002530_12_disp.npy', '002530_13_disp.npy', '002530_14_disp.npy', '002530_23_disp.npy', '002530_24_disp.npy', '002530_34_disp.npy']
  depthName = '002530_depth.npy'
  depthGT = np.load(os.path.join(root, 'depth', depthName))
  disps = []
  views = []
  axs = []
  ca2e = CA2ERP(512, 1024, 1024, 512, cuda=False)
  camPairs = ['12', '13', '14', '23', '24', '34']
  angles = ['0', '90', '45', '135', '90', '0']
  dsave = np.log(depthGT + 1.0)
  dsave = (dsave - np.min(dsave)) / (np.max(dsave) - np.min(dsave)) * 255
  dsave = cv2.applyColorMap(dsave.astype(np.uint8), cv2.COLORMAP_JET)
  cv2.imwrite(os.path.join(rootsave, 'depthGT' + '_' + name + '.png'), dsave)
  i = 0
  for cp in camPairs:
    disp = np.load(os.path.join(root, 'disp', dispNames[i]))
    mask_disp_is_0 = (disp == 0) | (np.isnan(disp))
    if i < 3:
      depth = torch.from_numpy(depthGT).unsqueeze_(0).unsqueeze_(0)
      d = ca2e.trans(depth, angles[i])
    else:
      depth = disp2depth(disp, i, keep_view=True)
      depth[mask_disp_is_0] = 0
      depth = torch.from_numpy(depth).unsqueeze_(0).unsqueeze_(0)
      d = ca2e.trans(depth, angles[i])
    d[d > 1000.0] = 1000.0
    dsave = d.squeeze(0).squeeze(0).numpy()
    dsave = np.log(dsave + 1.0)
    dsave = (dsave - np.min(dsave)) / (np.max(dsave) - np.min(dsave)) * 255
    dsave = cv2.applyColorMap(dsave.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(rootsave, 'depth' + '_' + cp + '.png'), dsave)
    #disp[mask_disp_is_0] = 0
    dsave = np.log(disp + 1.0)
    dsave = (dsave - np.min(dsave)) / (np.max(dsave) - np.min(dsave)) * 255
    dsave = cv2.applyColorMap(dsave.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(rootsave, 'disp' + '_' + cp + '.png'), dsave)

    # left = cv2.imread(os.path.join(root, 'rgb', name + '_' + cp + '_rgb' + cp[0] + '.png'))
    # right = cv2.imread(os.path.join(root, 'rgb', name + '_' + cp + '_rgb' + cp[1] + '.png'))
    # left = left.transpose((2, 0, 1)).astype(np.float32)
    # left = torch.from_numpy(left).unsqueeze_(0)
    # right = right.transpose((2, 0, 1)).astype(np.float32)
    # right = torch.from_numpy(right).unsqueeze_(0)
    # l, r = ca2e.transPairs(left, right, '0')
    # depthi = depthGtViewTrans(depthGT, i, depthGtTransConfigs)
    # depthi = torch.from_numpy(depthi).unsqueeze_(0).unsqueeze_(0)
    # d = ca2e.trans(depthi, '0')
    # lsave = l.squeeze(0).numpy().transpose((1, 2, 0))
    # rsave = r.squeeze(0).numpy().transpose((1, 2, 0))
    # dsave = d.squeeze(0).squeeze(0).numpy()

    # dsave = np.log(dsave + 1.0)
    # dsave = (dsave - np.min(dsave)) / (np.max(dsave) - np.min(dsave)) * 255
    # cv2.imwrite(os.path.join(rootsave, 'erp' + '_' + cp + '_' + cp[0] + '.png'), lsave)
    # cv2.imwrite(os.path.join(rootsave, 'erp' + '_' + cp + '_' + cp[1] + '.png'), rsave)
    # cv2.imwrite(os.path.join(rootsave, 'erpdepth' + '_' + cp + '.png'), dsave)
    i += 1

  #J = Cassini2Cassini(I, 0.5 * np.pi, 0, 0).astype(np.int)
  #print("J shape: {}, max: {}, min: {}".format(J.shape, np.max(J), np.min(J)))
  # for i in range(len(viewNames)):
  #   views.append(cv2.imread(os.path.join(root, 'rgb', viewNames[i])))
  #   axs.append(plt.subplot(2, 3, i + 1))
  #   axs[i].imshow(views[i])
  # plt.figure("rgb view trans")
  # for i in range(len(dispNames)):
  #   disps.append(np.load(os.path.join(root, 'disp', dispNames[i])))
  #   views.append(cv2.imread(os.path.join(root, 'rgb', viewNames[i])))
  #   axs.append(plt.subplot(2, 3, i + 1))
  #   disp = torch.from_numpy(disps[i]).unsqueeze_(0).unsqueeze_(0)
  #   s = time.time()
  #   depth = batchViewTransOri(disp, i, cuda=False)
  #   print(i, time.time() - s, torch.sum(torch.abs(depth - depthGT) > 0))
  #   depth = depth.squeeze_(0).squeeze_(0).numpy()

  #   axs[i].imshow(depth)
  # plt.show()
