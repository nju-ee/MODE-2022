import os

import numpy as np
import torch
import torch.nn.functional as F

# extrinsics of cameras
defaultConfigs = {
    'height': 1024,
    'width': 512,
    'cam_nums': 4,
    'extrinsics': [
        #y,z,x,pitch,yaw,roll
        #1
        [0,
         0,
         0,
         0,
         0,
         0],
        #2
        [0,
         0,
         -1,
         0,
         0,
         0],
        #3
        [0,
         -1,
         0,
         0,
         0,
         0],
        #4
        [0,
         -1,
         -1,
         0,
         0,
         0]
    ]
}


class CassiniSweepViewTrans():
  def __init__(self, configs=defaultConfigs, maxDepth=1000, minDepth=0.5, numInvs=192, scaleDown=2, numInvDown=2, device='cuda'):
    self.cam_nums = configs['cam_nums']
    self.height, self.width = configs['height'] // scaleDown, configs['width'] // scaleDown
    self.extrinsics = configs['extrinsics']
    self.maxDepth = maxDepth
    self.minDepth = minDepth
    self.numInvs = numInvs
    self.device = device
    self.scaleDown = scaleDown
    self.numInvDown = numInvDown

    self.min_invdepth = 1.0 / self.maxDepth
    self.max_invdepth = 1.0 / self.minDepth
    self.sample_step_invdepth = (self.max_invdepth - self.min_invdepth) / (self.numInvs - 1.0)
    self.invdepths = np.arange(self.min_invdepth, self.max_invdepth + self.sample_step_invdepth, self.sample_step_invdepth, dtype=np.float32)

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
    for i in range(self.cam_nums):
      y0, z0, x0, pitch, yaw, roll = self.extrinsics[i]
      self.t.append(torch.from_numpy(np.array([[x0], [y0], [z0]]).astype(np.float32)))
      Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
      Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
      Ry = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
      Ra = np.dot(np.dot(Rx, Rz), Ry)
      self.R.append(torch.from_numpy(np.linalg.inv(Ra).astype(np.float32)))

  def genCassiniSweepGrids(self):
    grids = []
    numInvs_2 = self.numInvs // self.numInvDown
    h, w = self.height, self.width
    for i in range(self.cam_nums):
      grid_i = torch.zeros((numInvs_2, h, w, 2), requires_grad=False).to(self.device)
      for d in range(numInvs_2):
        depth = 1.0 / self.invdepths[self.numInvDown * d]
        t0 = self.t[i].to(self.device)
        R0 = self.R[i].to(self.device)
        phi_l_map = torch.from_numpy(self.phi_1_map).unsqueeze_(0).to(self.device)
        theta_l_map = torch.from_numpy(self.theta_1_map).unsqueeze_(0).to(self.device)
        X1 = torch.cat([depth * torch.sin(phi_l_map), depth * torch.cos(phi_l_map) * torch.sin(theta_l_map), depth * torch.cos(phi_l_map) * torch.cos(theta_l_map)], 1).view(3, -1)
        X2 = torch.matmul(R0, (X1 - t0)).view(3, h, w)
        depth2 = torch.sqrt(X2[0, :, :]**2 + X2[1, :, :]**2 + X2[2, :, :]**2).unsqueeze_(0)
        depth2[depth2 > self.maxDepth] = self.maxDepth
        validMask2 = (depth2 > 0) & (~torch.isnan(depth2)) & (~torch.isinf(depth2)) & (depth2 <= self.maxDepth)
        theta_l_map2 = torch.zeros_like(depth2)
        phi_l_map2 = torch.zeros_like(depth2)
        theta_l_map2 = torch.atan2(X2[1:2, :, :], X2[2:, :, :])
        phi_l_map2[validMask2] = torch.asin(torch.clamp(X2[0:1, :, :][validMask2] / depth2[validMask2], -1, 1))
        theta_l_map2 = (theta_l_map2 - torch.min(theta_l_map2)) / (torch.max(theta_l_map2) - torch.min(theta_l_map2)) * 2 - 1
        phi_l_map2 = (phi_l_map2 - torch.min(phi_l_map2)) / (torch.max(phi_l_map2) - torch.min(phi_l_map2)) * 2 - 1
        grid_i[d, ::] = torch.cat([-phi_l_map2, -theta_l_map2], 0).permute(1, 2, 0)
      grids.append(grid_i.clone())
    return grids

  def invIndex2depth(self, invIndex, start_index=0):
    invDep = self.min_invdepth + (invIndex - start_index) * self.sample_step_invdepth
    depth = 1.0 / invDep
    invalidMask = (depth < self.minDepth) | (depth > self.maxDepth)
    return invDep, depth, invalidMask

  def trans(self, depth, feature, camNum):
    n, c, h, w = depth.shape
    n, c2, h, w = feature.shape
    camNum -= 1
    assert camNum >= 0 and camNum < self.cam_nums
    t0 = self.t[camNum].to(self.device)
    R0 = self.R[camNum].to(self.device)
    phi_l_map = torch.from_numpy(self.phi_1_map).unsqueeze_(0).unsqueeze_(0).to(self.device).repeat(n, 1, 1, 1)
    theta_l_map = torch.from_numpy(self.theta_1_map).unsqueeze_(0).unsqueeze_(0).to(self.device).repeat(n, 1, 1, 1)
    X1 = torch.cat([depth * torch.sin(phi_l_map), depth * torch.cos(phi_l_map) * torch.sin(theta_l_map), depth * torch.cos(phi_l_map) * torch.cos(theta_l_map)], 1).view(n, 3, -1)
    X2 = torch.matmul(R0, (X1 - t0.repeat(n, 1, 1))).view(n, 3, h, w)  # DEBUG 1->other (+); other->1(-)
    depth2 = torch.sqrt(X2[:, 0, :, :]**2 + X2[:, 1, :, :]**2 + X2[:, 2, :, :]**2).unsqueeze_(1)
    depth2[depth2 > self.maxDepth] = self.maxDepth
    validMask2 = (depth2 > 0) & (~torch.isnan(depth2)) & (~torch.isinf(depth2)) & (depth2 <= self.maxDepth)
    theta_l_map2 = torch.zeros_like(depth2)
    phi_l_map2 = torch.zeros_like(depth2)
    theta_l_map2 = torch.atan2(X2[:, 1:2, :, :], X2[:, 2:, :, :])
    phi_l_map2[validMask2] = torch.asin(torch.clamp(X2[:, 0:1, :, :][validMask2] / depth2[validMask2], -1, 1))
    theta_l_map2 = (theta_l_map2 - torch.min(theta_l_map2)) / (torch.max(theta_l_map2) - torch.min(theta_l_map2)) * 2 - 1
    phi_l_map2 = (phi_l_map2 - torch.min(phi_l_map2)) / (torch.max(phi_l_map2) - torch.min(phi_l_map2)) * 2 - 1
    grid = torch.cat([-phi_l_map2, -theta_l_map2], 1).permute(0, 2, 3, 1)
    featureTrans = F.grid_sample(feature, grid, mode='bilinear', align_corners=True, padding_mode='border')
    validMask2 = validMask2.repeat((1, c2, 1, 1))
    featureTrans[~validMask2] = 0  # set the depth values of invalid points to 0.
    return featureTrans


if __name__ == '__main__':
  import cv2
  CSVT = CassiniSweepViewTrans(scaleDown=1, device='cpu')
  grids = CSVT.genCassiniSweepGrids()
  print(len(grids), grids[0].shape)
  #TODO: test view trans and sweep
  rootDir = '../../datasets/Deep360/depth_on_1/ep1_500frames/testing'
  leftName = os.path.join(rootDir, 'rgb', '002505_12_rgb1.png')
  depthName = os.path.join(rootDir, 'depth', '002505_depth.npy')
  img1 = cv2.imread(leftName).astype(np.float32)
  depth1 = np.load(depthName).astype(np.float32)
  depth1 = torch.from_numpy(depth1).unsqueeze(0).unsqueeze(0)
  img1 = torch.from_numpy(img1.transpose((2, 0, 1))).unsqueeze(0)
  img2 = CSVT.trans(depth1, img1, 4)
  img2 = img2.squeeze(0).numpy().transpose((1, 2, 0))
  img2 = img2.astype(np.uint8)
  cv2.imwrite('ca_trans_4.png', img2)
