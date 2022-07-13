import torch
import torch.nn.functional as F
import numpy as np
from numba import jit

def cassini2Equirec(cassini):
    if cassini.ndim == 2:
        cassini = np.expand_dims(cassini, axis=-1)
        source_image = torch.FloatTensor(cassini).unsqueeze(0).transpose(1, 3).transpose(2, 3).cuda()
    elif cassini.ndim == 3:
        source_image = torch.FloatTensor(cassini).unsqueeze(0).transpose(1, 3).transpose(2, 3).cuda()
    else:
        source_image = cassini
  
    erp_h = source_image.shape[-1]
    erp_w = source_image.shape[-2]

    theta_erp_start = np.pi - (np.pi / erp_w)
    theta_erp_end = -np.pi
    theta_erp_step = 2 * np.pi / erp_w
    theta_erp_range = np.arange(theta_erp_start, theta_erp_end, -theta_erp_step)
    theta_erp_map = np.array([theta_erp_range for i in range(erp_h)]).astype(np.float32)

    phi_erp_start = 0.5 * np.pi - (0.5 * np.pi / erp_h)
    phi_erp_end = -0.5 * np.pi
    phi_erp_step = np.pi / erp_h
    phi_erp_range = np.arange(phi_erp_start, phi_erp_end, -phi_erp_step)
    phi_erp_map = np.array([phi_erp_range for j in range(erp_w)]).astype(np.float32).T

    theta_cassini_map = np.arctan2(np.tan(phi_erp_map), np.cos(theta_erp_map))
    phi_cassini_map = np.arcsin(np.cos(phi_erp_map) * np.sin(theta_erp_map))

    grid_x = torch.FloatTensor(np.clip(-phi_cassini_map / (0.5 * np.pi),-1,1)).unsqueeze(-1).cuda()
    grid_y = torch.FloatTensor(np.clip(-theta_cassini_map / np.pi,-1,1)).unsqueeze(-1).cuda()
    grid = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0).repeat_interleave(source_image.shape[0], dim=0)

    sampled_image = F.grid_sample(
                                    source_image, 
                                    grid,
                                    mode='bilinear',
                                    align_corners=True,
                                    padding_mode='border'
                                    ) # 1, ch, self.output_h, self.output_w

    if cassini.ndim == 3:
        erp = sampled_image.transpose(1, 3).transpose(1, 2).data.cpu().numpy()[0].astype(cassini.dtype)
        return erp.squeeze()
    else:
        erp = sampled_image
        return erp.squeeze(1)

def cassini2Cassini(cassini_1, pitch, yaw, roll):
  Rx = np.array([[1, 0, 0],
                                  [0, np.cos(roll), -np.sin(roll)],
                                  [0, np.sin(roll), np.cos(roll)]])

  Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                  [np.sin(yaw), np.cos(yaw), 0],
                                  [0, 0, 1]])
  
  Ry = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                                  [0, 1, 0],
                                  [np.sin(pitch), 0, np.cos(pitch)]])

  R=np.dot(np.dot(Rx,Rz),Ry)
  R_I=np.linalg.inv(R)

  output_h = cassini_1.shape[0]
  output_w = cassini_1.shape[1]

  theta_2_start = np.pi - (np.pi / output_h)
  theta_2_end = -np.pi
  theta_2_step = 2 * np.pi / output_h
  theta_2_range = np.arange(theta_2_start, theta_2_end, -theta_2_step)
  theta_2_map = np.array([theta_2_range for i in range(output_w)]).astype(np.float32).T

  phi_2_start = 0.5 * np.pi - (0.5 * np.pi / output_w)
  phi_2_end = -0.5 * np.pi
  phi_2_step = np.pi / output_w
  phi_2_range = np.arange(phi_2_start, phi_2_end, -phi_2_step)
  phi_2_map = np.array([phi_2_range for j in range(output_h)]).astype(np.float32)

  x_2 = np.sin(phi_2_map)
  y_2 = np.cos(phi_2_map)*np.sin(theta_2_map)
  z_2 = np.cos(phi_2_map)*np.cos(theta_2_map)
  X_2 = np.expand_dims(np.dstack((x_2,y_2,z_2)), axis=-1)

  X_1 = np.matmul(R_I, X_2)

  theta_1_map = np.arctan2(X_1[:,:,1,0], X_1[:,:,2,0])
  phi_1_map = np.arcsin(np.clip(X_1[:,:,0,0],-1,1))

  source_image = torch.FloatTensor(cassini_1).unsqueeze(0).transpose(1, 3).transpose(2, 3).cuda()
  grid_x = torch.FloatTensor(np.clip(-phi_1_map / (0.5 * np.pi),-1,1)).unsqueeze(-1).cuda()
  grid_y = torch.FloatTensor(np.clip(-theta_1_map / np.pi,-1,1)).unsqueeze(-1).cuda()
  grid = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0)

  sampled_image = F.grid_sample(
                                source_image, 
                                grid,
                                mode='bilinear',
                                align_corners=True,
                                padding_mode='border'
                                ) # 1, ch, self.output_h, self.output_w

  cassini_2 = sampled_image.transpose(1, 3).transpose(1, 2).data.cpu().numpy()[0].astype(cassini_1.dtype)
  return cassini_2

def cassini2Cassini_matrix(cassini_1, R):
    R_I = R.T # 旋转矩阵的转置即为它的逆矩阵

    output_h = cassini_1.shape[0]
    output_w = cassini_1.shape[1]

    theta_2_start = np.pi - (np.pi / output_h)
    theta_2_end = -np.pi
    theta_2_step = 2 * np.pi / output_h
    theta_2_range = np.arange(theta_2_start, theta_2_end, -theta_2_step)
    theta_2_map = np.array([theta_2_range for i in range(output_w)]).astype(np.float32).T

    phi_2_start = 0.5 * np.pi - (0.5 * np.pi / output_w)
    phi_2_end = -0.5 * np.pi
    phi_2_step = np.pi / output_w
    phi_2_range = np.arange(phi_2_start, phi_2_end, -phi_2_step)
    phi_2_map = np.array([phi_2_range for j in range(output_h)]).astype(np.float32)

    x_2 = np.sin(phi_2_map)
    y_2 = np.cos(phi_2_map)*np.sin(theta_2_map)
    z_2 = np.cos(phi_2_map)*np.cos(theta_2_map)
    X_2 = np.expand_dims(np.dstack((x_2,y_2,z_2)), axis=-1)

    X_1 = np.matmul(R_I, X_2)

    theta_1_map = np.arctan2(X_1[:,:,1,0], X_1[:,:,2,0])
    phi_1_map = np.arcsin(np.clip(X_1[:,:,0,0],-1,1))

    source_image = torch.FloatTensor(cassini_1).unsqueeze(0).transpose(1, 3).transpose(2, 3).cuda()
    grid_x = torch.FloatTensor(np.clip(-phi_1_map / (0.5 * np.pi),-1,1)).unsqueeze(-1).cuda()
    grid_y = torch.FloatTensor(np.clip(-theta_1_map / np.pi,-1,1)).unsqueeze(-1).cuda()
    grid = torch.cat([grid_x, grid_y], dim=-1).unsqueeze(0)

    sampled_image = F.grid_sample(source_image, grid, mode='bilinear', align_corners=True, padding_mode='border') # 1, ch, self.output_h, self.output_w

    cassini_2 = sampled_image.transpose(1, 3).transpose(1, 2).data.cpu().numpy()[0].astype(cassini_1.dtype)
    return cassini_2

def depthViewTrans(view_1, y0, z0, x0, pitch, yaw, roll):
  Rx = np.array([[1, 0, 0],
                                  [0, np.cos(roll), -np.sin(roll)],
                                  [0, np.sin(roll), np.cos(roll)]])

  Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                  [np.sin(yaw), np.cos(yaw), 0],
                                  [0, 0, 1]])
  
  Ry = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                                  [0, 1, 0],
                                  [np.sin(pitch), 0, np.cos(pitch)]])

  R=np.dot(np.dot(Rx,Rz),Ry)

  t=np.array([[x0],[y0],[z0]])

  output_h = view_1.shape[0]
  output_w = view_1.shape[1]

  theta_1_start = np.pi - (np.pi / output_h)
  theta_1_end = -np.pi
  theta_1_step = 2 * np.pi / output_h
  theta_1_range = np.arange(theta_1_start, theta_1_end, -theta_1_step)
  theta_1_map = np.array([theta_1_range for i in range(output_w)]).astype(np.float32).T

  phi_1_start = 0.5 * np.pi - (0.5 * np.pi / output_w)
  phi_1_end = -0.5 * np.pi
  phi_1_step = np.pi / output_w
  phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
  phi_1_map = np.array([phi_1_range for j in range(output_h)]).astype(np.float32)

  r_1 = view_1

  x_1 = r_1 * np.sin(phi_1_map)
  y_1 = r_1 * np.cos(phi_1_map)*np.sin(theta_1_map)
  z_1 = r_1 * np.cos(phi_1_map)*np.cos(theta_1_map)
  X_1 = np.expand_dims(np.dstack((x_1,y_1,z_1)), axis=-1)

  X_2 = np.matmul(R, X_1 - t)

  r_2 = np.sqrt(np.square(X_2[:,:,0,0]) + np.square(X_2[:,:,1,0]) + np.square(X_2[:,:,2,0]))
  theta_2_map = np.arctan2(X_2[:,:,1,0], X_2[:,:,2,0])
  phi_2_map = np.arcsin(np.clip(X_2[:,:,0,0]/r_2,-1,1))

  view_2=np.ones((output_h,output_w)).astype(np.float32)*100000

  I_2 = np.clip(np.rint(output_h / 2 - output_h * theta_2_map / (2 * np.pi)),0,output_h-1).astype(np.int16)
  J_2 = np.clip(np.rint(output_w / 2 - output_w * phi_2_map / np.pi),0,output_w-1).astype(np.int16)

  view_2 = __iterPixels(output_h, output_w, r_1, r_2, view_2, I_2, J_2)
  
  view_2[view_2==100000]=0
  view_2 = view_2.astype(np.float32)
  view_2[view_2 > 1000] = 1000

  return view_2

def depthViewTrans_matrix(view_1, R, t):
  output_h = view_1.shape[0]
  output_w = view_1.shape[1]

  theta_1_start = np.pi - (np.pi / output_h)
  theta_1_end = -np.pi
  theta_1_step = 2 * np.pi / output_h
  theta_1_range = np.arange(theta_1_start, theta_1_end, -theta_1_step)
  theta_1_map = np.array([theta_1_range for i in range(output_w)]).astype(np.float32).T

  phi_1_start = 0.5 * np.pi - (0.5 * np.pi / output_w)
  phi_1_end = -0.5 * np.pi
  phi_1_step = np.pi / output_w
  phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
  phi_1_map = np.array([phi_1_range for j in range(output_h)]).astype(np.float32)

  r_1 = view_1

  x_1 = r_1 * np.sin(phi_1_map)
  y_1 = r_1 * np.cos(phi_1_map)*np.sin(theta_1_map)
  z_1 = r_1 * np.cos(phi_1_map)*np.cos(theta_1_map)
  X_1 = np.expand_dims(np.dstack((x_1,y_1,z_1)), axis=-1)

  X_2 = R @ X_1 + t

  r_2 = np.sqrt(np.square(X_2[:,:,0,0]) + np.square(X_2[:,:,1,0]) + np.square(X_2[:,:,2,0]))
  theta_2_map = np.arctan2(X_2[:,:,1,0], X_2[:,:,2,0])
  phi_2_map = np.arcsin(np.clip(X_2[:,:,0,0]/r_2,-1,1))

  view_2=np.ones((output_h,output_w)).astype(np.float32)*100000

  I_2 = np.clip(np.rint(output_h / 2 - output_h * theta_2_map / (2 * np.pi)),0,output_h-1).astype(np.int16)
  J_2 = np.clip(np.rint(output_w / 2 - output_w * phi_2_map / np.pi),0,output_w-1).astype(np.int16)

  view_2 = __iterPixels(output_h, output_w, r_1, r_2, view_2, I_2, J_2)
  
  view_2[view_2==100000]=0
  view_2 = view_2.astype(np.float32)
  view_2[view_2 > 1000] = 1000

  return view_2

@jit(nopython = True)
def __iterPixels(output_h, output_w, r_1, r_2, view_2, I_2, J_2):
    for i in range(output_h):
        for j in range(output_w):
            if r_1[i,j] > 0:
                flag = r_2[i,j] < view_2[I_2[i,j],J_2[i,j]]
                view_2[I_2[i,j],J_2[i,j]] = flag*r_2[i,j]+(1 - flag)*view_2[I_2[i,j],J_2[i,j]]
    return view_2

def depthViewTransWithConf(view_1, conf_1, y0, z0, x0, pitch, yaw, roll):
  Rx = np.array([[1, 0, 0],
                                  [0, np.cos(roll), -np.sin(roll)],
                                  [0, np.sin(roll), np.cos(roll)]])

  Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                  [np.sin(yaw), np.cos(yaw), 0],
                                  [0, 0, 1]])
  
  Ry = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                                  [0, 1, 0],
                                  [np.sin(pitch), 0, np.cos(pitch)]])

  R=np.dot(np.dot(Rx,Rz),Ry)

  t=np.array([[x0],[y0],[z0]])

  output_h = view_1.shape[0]
  output_w = view_1.shape[1]

  theta_1_start = np.pi - (np.pi / output_h)
  theta_1_end = -np.pi
  theta_1_step = 2 * np.pi / output_h
  theta_1_range = np.arange(theta_1_start, theta_1_end, -theta_1_step)
  theta_1_map = np.array([theta_1_range for i in range(output_w)]).astype(np.float32).T

  phi_1_start = 0.5 * np.pi - (0.5 * np.pi / output_w)
  phi_1_end = -0.5 * np.pi
  phi_1_step = np.pi / output_w
  phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
  phi_1_map = np.array([phi_1_range for j in range(output_h)]).astype(np.float32)

  r_1 = view_1

  x_1 = r_1 * np.sin(phi_1_map)
  y_1 = r_1 * np.cos(phi_1_map)*np.sin(theta_1_map)
  z_1 = r_1 * np.cos(phi_1_map)*np.cos(theta_1_map)
  X_1 = np.expand_dims(np.dstack((x_1,y_1,z_1)), axis=-1)

  X_2 = np.matmul(R, X_1 - t)

  r_2 = np.sqrt(np.square(X_2[:,:,0,0]) + np.square(X_2[:,:,1,0]) + np.square(X_2[:,:,2,0]))
  theta_2_map = np.arctan2(X_2[:,:,1,0], X_2[:,:,2,0])
  phi_2_map = np.arcsin(np.clip(X_2[:,:,0,0]/r_2,-1,1))

  view_2=np.ones((output_h,output_w)).astype(np.float32)*100000
  conf_2 = np.zeros((output_h,output_w)).astype(np.float32)

  I_2 = np.clip(np.rint(output_h / 2 - output_h * theta_2_map / (2 * np.pi)),0,output_h-1).astype(np.int16)
  J_2 = np.clip(np.rint(output_w / 2 - output_w * phi_2_map / np.pi),0,output_w-1).astype(np.int16)

  view_2, conf_2 = __iterPixels_with_conf(output_h, output_w, conf_1, conf_2, r_1, r_2, view_2, I_2, J_2)

  view_2[view_2==100000]=0
  view_2 = view_2.astype(np.float32)
  view_2[view_2 > 1000] = 1000
  
  return view_2, conf_2

def depthViewTransWithConf_matrix(view_1, conf_1, R, t):
  output_h = view_1.shape[0]
  output_w = view_1.shape[1]

  theta_1_start = np.pi - (np.pi / output_h)
  theta_1_end = -np.pi
  theta_1_step = 2 * np.pi / output_h
  theta_1_range = np.arange(theta_1_start, theta_1_end, -theta_1_step)
  theta_1_map = np.array([theta_1_range for i in range(output_w)]).astype(np.float32).T

  phi_1_start = 0.5 * np.pi - (0.5 * np.pi / output_w)
  phi_1_end = -0.5 * np.pi
  phi_1_step = np.pi / output_w
  phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
  phi_1_map = np.array([phi_1_range for j in range(output_h)]).astype(np.float32)

  r_1 = view_1

  x_1 = r_1 * np.sin(phi_1_map)
  y_1 = r_1 * np.cos(phi_1_map)*np.sin(theta_1_map)
  z_1 = r_1 * np.cos(phi_1_map)*np.cos(theta_1_map)
  X_1 = np.expand_dims(np.dstack((x_1,y_1,z_1)), axis=-1)

  X_2 = R @ X_1 + t

  r_2 = np.sqrt(np.square(X_2[:,:,0,0]) + np.square(X_2[:,:,1,0]) + np.square(X_2[:,:,2,0]))
  theta_2_map = np.arctan2(X_2[:,:,1,0], X_2[:,:,2,0])
  phi_2_map = np.arcsin(np.clip(X_2[:,:,0,0]/r_2,-1,1))

  view_2=np.ones((output_h,output_w)).astype(np.float32)*100000
  conf_2 = np.zeros((output_h,output_w)).astype(np.float32)

  I_2 = np.clip(np.rint(output_h / 2 - output_h * theta_2_map / (2 * np.pi)),0,output_h-1).astype(np.int16)
  J_2 = np.clip(np.rint(output_w / 2 - output_w * phi_2_map / np.pi),0,output_w-1).astype(np.int16)

  view_2, conf_2 = __iterPixels_with_conf(output_h, output_w, conf_1, conf_2, r_1, r_2, view_2, I_2, J_2)

  view_2[view_2==100000]=0
  view_2 = view_2.astype(np.float32)
  view_2[view_2 > 1000] = 1000
  
  return view_2, conf_2

@jit(nopython = True)
def __iterPixels_with_conf(output_h, output_w, conf_1, conf_2, r_1, r_2, view_2, I_2, J_2):
    for i in range(output_h):
        for j in range(output_w):
            if r_1[i,j] > 0:
                flag = r_2[i,j] < view_2[I_2[i,j],J_2[i,j]]
                view_2[I_2[i,j],J_2[i,j]] = flag*r_2[i,j]+(1 - flag)*view_2[I_2[i,j],J_2[i,j]]
                conf_2[I_2[i,j],J_2[i,j]] = flag*conf_1[i,j]+(1 - flag)*conf_2[I_2[i,j],J_2[i,j]]
    return view_2, conf_2