import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import random
import numpy as np
import math
from multiprocessing import Pool, sharedctypes
import time


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
    # Lock(view_2[I_2[i,j],J_2[i,j]])
    flag = r_2[i][j] < view_2[I_2[i][j]][J_2[i][j]]
    view_2[I_2[i][j]][J_2[i][j]] = flag * r_2[i][j] + (1 - flag) * view_2[I_2[i][j]][J_2[i][j]]
    # unLock(view_2[I_2[i,j],J_2[i,j]])


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


def disp2depth(disp, cam_pair, max_depth=1000, parallel=True):
  if parallel:
    transFunc = View_trans_parallel
  else:
    transFunc = View_trans
  B = np.array([1, 1, math.sqrt(2), math.sqrt(2), 1, 1]).astype(np.float32)

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

  if cam_pair == 0:
    return depth_l
  elif cam_pair == 1:
    cassini_1 = np.expand_dims(depth_l, axis=-1)
    cassini_2 = Cassini2Cassini(cassini_1, 0.5 * math.pi, 0, 0)
    return cassini_2[:, :, 0]
  elif cam_pair == 2:
    cassini_1 = np.expand_dims(depth_l, axis=-1)
    cassini_2 = Cassini2Cassini(cassini_1, 0.25 * math.pi, 0, 0)
    return cassini_2[:, :, 0]
  elif cam_pair == 3:
    view_2 = transFunc(depth_l, 0, -math.sqrt(2) / 2, -math.sqrt(2) / 2, 0.75 * math.pi, 0, 0)
    return view_2
  elif cam_pair == 4:
    view_2 = transFunc(depth_l, 0, -1, 0, 0.5 * math.pi, 0, 0)
    return view_2
  elif cam_pair == 5:
    view_2 = transFunc(depth_l, 0, 1, 0, 0, 0, 0)
    return view_2
  else:
    return None
