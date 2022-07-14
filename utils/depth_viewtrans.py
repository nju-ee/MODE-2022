import os
import math
import numpy as np
import time
from numba import jit

def View_trans(view_1, y0, z0, x0, pitch, yaw, roll):
  Rx = np.array([[1, 0, 0],
                                  [0, math.cos(roll), -math.sin(roll)],
                                  [0, math.sin(roll), math.cos(roll)]])

  Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                  [math.sin(yaw), math.cos(yaw), 0],
                                  [0, 0, 1]])
  
  Ry = np.array([[math.cos(pitch), 0, -math.sin(pitch)],
                                  [0, 1, 0],
                                  [math.sin(pitch), 0, math.cos(pitch)]])

  R=np.dot(np.dot(Rx,Rz),Ry)

  t=np.array([[x0],[y0],[z0]])

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
  y_1 = r_1 * np.cos(phi_1_map)*np.sin(theta_1_map)
  z_1 = r_1 * np.cos(phi_1_map)*np.cos(theta_1_map)
  X_1 = np.expand_dims(np.dstack((x_1,y_1,z_1)), axis=-1)

  X_2 = np.matmul(R, X_1 - t)

  r_2 = np.sqrt(np.square(X_2[:,:,0,0]) + np.square(X_2[:,:,1,0]) + np.square(X_2[:,:,2,0]))
  theta_2_map = np.arctan2(X_2[:,:,1,0], X_2[:,:,2,0])
  phi_2_map = np.arcsin(np.clip(X_2[:,:,0,0]/r_2,-1,1))

  view_2=np.ones((output_h,output_w)).astype(np.float32)*100000

  I_2 = np.clip(np.rint(output_h / 2 - output_h * theta_2_map / (2 * math.pi)),0,output_h-1).astype(np.int16)
  J_2 = np.clip(np.rint(output_w / 2 - output_w * phi_2_map / math.pi),0,output_w-1).astype(np.int16)

  view_2 = iter_pixels(output_h, output_w, r_2, view_2, I_2, J_2)
  
  view_2[view_2==100000]=0
  view_2 = view_2.astype(np.float32)

  return view_2

@jit(nopython = True)
def iter_pixels(output_h, output_w, r_2, view_2, I_2, J_2):
  for i in range(output_h):
    for j in range(output_w):
      flag = r_2[i,j] < view_2[I_2[i,j],J_2[i,j]]
      view_2[I_2[i,j],J_2[i,j]] = flag*r_2[i,j]+(1 - flag)*view_2[I_2[i,j],J_2[i,j]]
  return view_2

if __name__ == "__main__":
  rootdir = "./view1/"
  outdir = "./view2/"

  name = "000499_23_depth_pred.npy"
  view_1 = np.load(rootdir + name).astype(np.float32)

  start_time = time.time()

  view_2 = View_trans(view_1, 0, -math.sqrt(2)/2, -math.sqrt(2)/2, 0.75 * math.pi, 0, 0)
  # view_2 = View_trans(view_1, 0, -1, 0, 0.5 * math.pi, 0, 0)
  # view_2=View_trans_vect(view_1, 0, 1, 0, 0, 0, 0)

  print('time = %.3f s' %(time.time() - start_time))

  np.save(outdir + name[:-4]+ '_trans.npy', view_2)