import os
import cv2
import open3d
import numpy as np

from utils.ERPandCassini import CA2ERP

rgbName = '../tmp/B_58_086_12_rgb1_erp.png'
depthName = '../tmp/B_58_086_depth_pred_erp.npy'
rgb = cv2.imread(rgbName)
depth = np.load(depthName)
cv2.imwrite(depthName[:-3] + 'exr', depth)
rgb = rgb.astype(np.float64)
depth
rgb = rgb / 255
print(rgb.shape, depth.shape)
h, w = rgb.shape[:2]
colors = []
positions = []
allPoints = []
for i in range(h):
  for j in range(w):
    d = depth[i, j]
    if d < 0: continue
    color = rgb[i, j, :]
    colors.append(color)
    theta = i / h * np.pi - np.pi / 2.0
    phi = j / w * (2 * np.pi) - 3.0 * np.pi / 2.0
    y = (d) * np.sin(theta)
    x = (d) * np.cos(theta) * np.cos(phi)
    z = (d) * np.cos(theta) * np.sin(phi)
    positions.append([x, y, z])
colors = np.array(colors)
positions = np.array(positions).squeeze(-1)
print(colors.shape, positions.shape)
o3dcolors = open3d.utility.Vector3dVector(colors)
o3dpoints = open3d.utility.Vector3dVector(positions)
pcd = open3d.geometry.PointCloud()
pcd.points = o3dpoints
pcd.colors = o3dcolors
# rgb = open3d.cpu.pybind.core.Tensor.from_numpy(rgb)
# depth = open3d.cpu.pybind.core.Tensor.from_numpy(depth)
#rgbdImg = open3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
# pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbdImg, open3d.camera.PinholeCameraIntrinsic(open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# # Flip it, otherwise the pointcloud will be upside down
# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
open3d.visualization.draw_geometries([pcd])