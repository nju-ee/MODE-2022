import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import random
from PIL import Image
import numpy as np
import cv2

from . import preprocess


def default_loader(path):
  return Image.open(path).convert('RGB')


def disparity_loader(path):
  return np.load(path)['arr_0'].astype(np.float32)


def depth_loader(path):
  depth = np.load(path)['arr_0'].astype(np.float32)
  return np.expand_dims(depth, axis=-1)


def conf_loader(path):
  conf = cv2.imread(path)
  return np.expand_dims((conf[:, :, 0] / 255.0).astype(np.float32), axis=0)


"""
NOTE:

Multi-view 360-degree Dataset;
3000 frames;
each frame contains 6 pairs of left-right cassini preojection images
(12,13,14,23,24,34), 6 disparity maps and 1 depth maps referenced to the coordinate system of camera 1;

Dataset directory structure:
Deep360
+-- README.txt
+-- ep1_500frames
|   +-- training (350 frames)
|   |   +-- rgb (each frame consists of 6 pairs of rectified panoramas)
|   |   +-- rgb_soiled (soiled panoramas)
|   |   +-- disp (each frame consists of 6 disparity maps)
|   |   +-- depth (each frame consists of 1 ground truth depth map)
|   +-- validation (50 frames)
|   +-- testing (100 frames)
+-- ep2_500frames
+-- ep3_500frames
+-- ep4_500frames
+-- ep5_500frames
+-- ep6_500frames

"""


class Deep360DatasetDisparity(Dataset):
  def __init__(self, leftImgs, rightImgs, disps, shape=(1024, 512), crop=False, disploader=disparity_loader, rgbloader=default_loader):
    # NOTE：Arguments
    # leftImgs: a list contains file names of left images
    # rightImgs: a list contains file names of right images
    # disps: a list contains file names of groundtruth disparity maps
    # shape: img shape of input data. a tuple of (height, width)
    # crop: Bool. if set True, will use random crop
    # disploader: function to load dispairity maps
    # rgbloader: function to load left and right RGB images

    # Initialization
    super(Deep360DatasetDisparity, self).__init__()

    # Member variable assignment
    self.crop = crop
    self.height, self.width = shape
    self.processed = preprocess.get_transform_stage1(augment=False)  # transform of rgb images
    self.leftImgs = leftImgs
    self.rightImgs = rightImgs
    self.disps = disps
    self.disp_loader = disploader
    self.rgb_loader = rgbloader

  def __getitem__(self, index):
    inputs = {}
    leftName = self.leftImgs[index]
    rightName = self.rightImgs[index]
    dispName = self.disps[index]
    left = self.rgb_loader(leftName)
    right = self.rgb_loader(rightName)
    disp = self.disp_loader(dispName)
    # if need to resize
    w, h = left.size
    if (w != self.width):
      left = left.resize((self.width, self.height))
      right = right.resize((self.width, self.height))
      disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_NEAREST) * (self.width / w)
    # if random crop
    if self.crop:
      w, h = left.size
      th, tw = 512, 256
      x1 = random.randint(0, w - tw)
      y1 = random.randint(0, h - th)
      leftImg = leftImg.crop((x1, y1, x1 + tw, y1 + th))
      rightImg = rightImg.crop((x1, y1, x1 + tw, y1 + th))
      dispMap = dispMap[y1:y1 + th, x1:x1 + tw]
    disp = np.ascontiguousarray(disp, dtype=np.float32)
    left = self.processed(left)
    right = self.processed(right)
    disp = torch.from_numpy(disp).unsqueeze_(0)
    inputs = {'leftImg': left, 'rightImg': right, 'dispMap': disp, 'dispNames': dispName}
    return inputs

  def __len__(self):
    return len(self.disps)


class Deep360DatasetFusion(Dataset):
  def __init__(self, depthes, confs, rgbs, gt, resize, training, depthloader=depth_loader, rgbloader=default_loader):
    # Initialization
    super(Deep360DatasetFusion, self).__init__()
    self.depthes = depthes
    self.confs = confs
    self.rgbs = rgbs
    self.gt = gt
    self.depthloader = depthloader
    self.rgbloader = rgbloader
    self.resize = resize
    self.training = training

  def __getitem__(self, index):
    depthes = []
    confs = []
    rgbs = []

    for depth in self.depthes:
      depthes.append(self.depthloader(depth[index]))
    for conf in self.confs:
      confs.append(conf_loader(conf[index]))
    for rgb in self.rgbs:
      rgbs.append(self.rgbloader(rgb[index]))
    gt = self.depthloader(self.gt[index])
    gt = np.squeeze(gt, axis=-1)
    gt = np.ascontiguousarray(gt, dtype=np.float32)

    if self.resize:
      for i, depth in enumerate(depthes):
        depthes[i] = depth[::2, ::2, :]
      for i, conf in enumerate(confs):
        confs[i] = conf[:, ::2, ::2]
      w, h = rgbs[0].size
      for i, rgb in enumerate(rgbs):
        rgbs[i] = rgb.resize((int(w / 2), int(h / 2)))
      if self.training:
        gt = gt[::2, ::2]

    processed = preprocess.get_transform_stage2(augment=False)
    for i, depth in enumerate(depthes):
      depthes[i] = processed(depth)

    processed = preprocess.get_transform_stage1(augment=False)
    for i, rgb in enumerate(rgbs):
      rgbs[i] = processed(rgb)

    return self.gt[index], depthes, confs, rgbs, gt

  def __len__(self):
    return len(self.depthes[0])