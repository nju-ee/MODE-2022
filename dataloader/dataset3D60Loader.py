###################################
# 360 dataset pytorch dataloader
###################################
import os
import sys
import pickle

import numpy as np
import cv2
import PIL.Image as Image
import datetime
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
# Ignore warnings
import warnings

import torchvision

warnings.filterwarnings("ignore")

sys.path.append("../")
from utils.geometry import erp2rect_cassini
if __name__ == '__main__':
  import preprocess
else:
  from . import preprocess
#from meshcnn.utils import interp_r2tos2

############################################################################################################
# We use a text file to hold our dataset's filenames
# The "filenames list" has the following format
#
# path/to/Left/rgb.png path/to/Right/rgb.png path/to/Up/rgb.png path/to/Left/depth.exr path/to/Right/depth.exr path/to/Up/depth.exr
#
# We also have a Trinocular version, but you get the feeling.
#############################################################################################################

scenes = ['ep1_500frames', 'ep2_500frames', 'ep3_500frames', 'ep4_500frames', 'ep5_500frames', 'ep6_500frames']
splits = ['training', 'testing', 'validation']
dataType = ['rgb', 'disp', 'depth']
soiledType = ['glare', 'mud', 'water']
soiledNum = ['1_soiled_cam', '2_soiled_cam']
spotNum = ['2_spot', '3_spot', '4_spot', '5_spot', '6_spot']
percent = ['05percent', '10percent', '15percent', '20percent']

dataModes = ['disparity', 'fusion', 'intermedia_fusion']
stereo_pairs = ['lr', 'ud', 'ur', 'all']  # left-right, up-down, up-right
# inputCamPairs = ['12', '21', '12', '21', '12', '21']
inputCamPairs = ['12']
inputRGBImgIds = [0, 1]


class Dataset3D60Disparity(Dataset):
  #360D Dataset#
  def __init__(self,
               filenamesFile,
               rootDir='../../datasets/3D60/',
               curStage='training',
               shape=(512,
                      256),
               crop=False,
               pair='lr',
               flip=False,
               maxDepth=20.0):  # 3D60 is a indoor dataset and set max depth as 20 meters
    #########################################################################################################
    # Arguments:
    # -filenamesFile: Absolute path to the aforementioned filenames .txt file
    # -transform    : (Optional) transform to be applied on a sample
    # -mode         : Dataset mode. Available options: mono, lr (Left-Right), ud (Up-Down), tc (Trinocular)
    # -dataType     : type of input imgs. 'erp' = Equirectangular projection, 'sphere' = s2 signal, 'all' = both type
    #########################################################################################################
    # Initialization
    super(Dataset3D60Disparity, self).__init__()
    # Assertion
    assert curStage in splits
    assert (rootDir is not None) and (rootDir != '')
    assert pair in stereo_pairs
    # Member variable assignment
    self.rootDir = rootDir
    self.curStage = curStage
    self.height, self.width = shape
    self.pair = pair
    self.crop = crop

    self.flip = flip

    self.filenamesFile = filenamesFile
    self.baseline = 0.26  # left-right baseline
    self.maxDepth = maxDepth

    self.prefix_l = os.path.join(self.rootDir, 'Center_Left_Down/')
    self.prefix_r = os.path.join(self.rootDir, 'Right/')
    self.prefix_u = os.path.join(self.rootDir, 'Up/')

    # self.prefixInter_l = os.path.join(self.interDir, 'Center_Left_Down/')
    # self.prefixInter_r = os.path.join(self.interDir, 'Right/')
    # self.prefixInter_u = os.path.join(self.interDir, 'Up/')

    self.processed = preprocess.get_transform_stage1(augment=False)  # transform of rgb images

    # self.cddt = CassiniDepthDispTransformer(height=self.height, width=self.width, maxDisp, maxDepth, baseline, device='cuda')

    # get file names
    self.fileNameList = self.__getFileList()
    self.phiMap = self.__genCassiniPhiMap()

    print("Dataset 3D60: Multi-views fish eye dataset. File list: {}. Num of files: {}. root dir: {}.".format(self.filenamesFile, len(self.fileNameList), self.rootDir))

  def __len__(self):
    return len(self.fileNameList)

  def __getFileList(self):
    fileNameList = []
    with open(self.filenamesFile) as f:
      lines = f.readlines()
      for line in lines:
        fileNameList.append(line.strip().split(" "))  # split by space
    return fileNameList

  def __getitem__(self, index):  #return data in disparity estimation task
    assert (self.fileNameList is not None) and (len(self.fileNameList) > 0)
    name = self.fileNameList[index]
    # left/down
    leftName = os.path.join(self.prefix_l, name[0][2:])
    leftDepthName = os.path.join(self.prefix_l, name[3][2:])
    # right
    rightName = os.path.join(self.prefix_r, name[1][2:])
    rightDepthName = os.path.join(self.prefix_r, name[4][2:])
    # up
    upName = os.path.join(self.prefix_u, name[2][2:])
    upDepthName = os.path.join(self.prefix_u, name[5][2:])

    if self.pair == 'lr':
      left = leftName
      right = rightName
      depth = leftDepthName
      depth_r = rightDepthName
      rotate_vector = np.array([0, 0, 0]).astype(np.float32)
    elif self.pair == 'ud':
      left = upName
      right = leftName
      depth = upDepthName
      depth_r = leftDepthName
      rotate_vector = np.array([0, 0, -np.pi / 2]).astype(np.float32)
    elif self.pair == 'ur':
      left = upName
      right = rightName
      depth = upDepthName
      depth_r = rightDepthName
      rotate_vector = np.array([0, 0, -np.pi / 4]).astype(np.float32)
    elif self.pair == 'all':
      # all means random select from lr/ud/ur
      ra = random.random()
      if ra < 1 / 3:  # lr
        left = leftName
        right = rightName
        depth = leftDepthName
        depth_r = rightDepthName
        rotate_vector = np.array([0, 0, 0]).astype(np.float32)
      elif 1 / 2 <= ra < 2 / 3:  # ud
        left = upName
        right = leftName
        depth = upDepthName
        depth_r = leftDepthName
        rotate_vector = np.array([0, 0, -np.pi / 2]).astype(np.float32)
      else:  #ur
        left = upName
        right = rightName
        depth = upDepthName
        depth_r = rightDepthName
        rotate_vector = np.array([0, 0, -np.pi / 4]).astype(np.float32)
    R = cv2.Rodrigues(rotate_vector)[0]

    leftRGB = np.array(Image.open(left).convert('RGB'))
    rightRGB = np.array(Image.open(right).convert('RGB'))
    leftRGB = erp2rect_cassini(leftRGB, R, self.height, self.width, devcice='cpu').astype(np.uint8)
    rightRGB = erp2rect_cassini(rightRGB, R, self.height, self.width, devcice='cpu').astype(np.uint8)

    leftDepth = np.array(cv2.imread(depth, cv2.IMREAD_ANYDEPTH)).astype(np.float32)
    rightDepth = np.array(cv2.imread(depth_r, cv2.IMREAD_ANYDEPTH)).astype(np.float32)
    leftDepth = erp2rect_cassini(leftDepth, R, self.height, self.width, devcice='cpu')
    rightDepth = erp2rect_cassini(rightDepth, R, self.height, self.width, devcice='cpu')

    leftRGB = cv2.resize(leftRGB, (self.width, self.height))
    rightRGB = cv2.resize(rightRGB, (self.width, self.height))
    leftDepth = cv2.resize(leftDepth, (self.width, self.height))
    rightDepth = cv2.resize(rightDepth, (self.width, self.height))

    leftImg, rightImg, depthMap = leftRGB, rightRGB, leftDepth
    leftImg_flip, rightImg_flip, depthMap_flip = cv2.flip(rightRGB, 1), cv2.flip(leftRGB, 1), cv2.flip(rightDepth, 1)

    depthMap[depthMap > self.maxDepth] = 0.0
    depthMap_flip[depthMap_flip > self.maxDepth] = 0.0

    # leftRGB[leftRGB > 255] = 255
    # leftRGB[leftRGB < 0] = 0
    # rightRGB[rightRGB > 255] = 255
    # rightRGB[rightRGB < 0] = 0
    leftImg = leftImg.astype(np.uint8)
    rightImg = rightImg.astype(np.uint8)
    leftImg_flip = leftImg_flip.astype(np.uint8)
    rightImg_flip = rightImg_flip.astype(np.uint8)

    # print(np.max(leftImg), np.min(leftImg))

    dispMap = self.__depth2disp(depthMap)
    dispMap_flip = self.__depth2disp(depthMap_flip)

    if self.crop:
      leftImg = Image.fromarray(leftImg.astype(np.uint8))
      rightImg = Image.fromarray(rightImg.astype(np.uint8))
      w, h = leftImg.size
      th, tw = self.height // 2, self.width // 2

      x1 = random.randint(0, w - tw)
      y1 = random.randint(0, h - th)

      leftImg = leftImg.crop((x1, y1, x1 + tw, y1 + th))
      rightImg = rightImg.crop((x1, y1, x1 + tw, y1 + th))

      dispMap = dispMap[y1:y1 + th, x1:x1 + tw]

      leftImg = self.processed(leftImg)
      rightImg = self.processed(rightImg)

      dispMap = torch.from_numpy(dispMap).unsqueeze_(0)
      data = {'leftImg': leftImg, 'rightImg': rightImg, 'dispMap': dispMap, 'leftNames': leftName}
    else:
      leftImg = self.processed(leftImg)
      rightImg = self.processed(rightImg)
      dispMap = torch.from_numpy(dispMap).unsqueeze_(0)
      leftImg_flip = self.processed(leftImg_flip)
      rightImg_flip = self.processed(rightImg_flip)
      dispMap_flip = torch.from_numpy(dispMap_flip).unsqueeze_(0)
      data = {
          'leftImg': leftImg,
          'rightImg': rightImg,
          'dispMap': dispMap,
          'leftImg_flip': leftImg_flip,
          'rightImg_flip': rightImg_flip,
          'dispMap_flip': dispMap_flip,
          'leftNames': left,
          'rightNames': right
      }
    return data

  def __genCassiniPhiMap(self):
    phi_l_start = 0.5 * np.pi - (0.5 * np.pi / self.width)
    phi_l_end = -0.5 * np.pi
    phi_l_step = np.pi / self.width
    phi_l_range = np.arange(phi_l_start, phi_l_end, -phi_l_step)
    phi_l_map = np.array([phi_l_range for j in range(self.height)]).astype(np.float32)
    return phi_l_map

  def __depth2disp(self, depthMap):
    mask_depth_0 = depthMap == 0
    invMask = (depthMap <= 0) | (depthMap > self.maxDepth)
    depth_not_0 = np.ma.array(depthMap, mask=invMask)
    phi_l_map = self.phiMap
    disp = self.width * (np.arcsin(
        np.clip(
            (depth_not_0 * np.sin(phi_l_map) + self.baseline) / np.sqrt(depth_not_0 * depth_not_0 + self.baseline * self.baseline - 2 * depth_not_0 * self.baseline * np.cos(phi_l_map + np.pi / 2)),
            -1,
            1)) - phi_l_map) / np.pi
    disp = disp.filled(np.nan)
    disp[disp < 0] = 0
    return disp


class Dataset3D60Fusion_2view(Dataset):
  #360D Dataset#
  def __init__(
      self,
      filenamesFile,
      rootDir='../../datasets/3D60/',  #rgb, disp,depth 
      inputDir='',  #depth from disparity stage
      curStage='training',
      shape=(512, 256),
      pair='lr',
      maxDepth=20.0,
      view='Center_Left_Down/'):  # 3D60 is a indoor dataset and set max depth as 20 meters
    #########################################################################################################
    # Arguments:
    # -filenamesFile: Absolute path to the aforementioned filenames .txt file
    # -transform    : (Optional) transform to be applied on a sample
    # -mode         : Dataset mode. Available options: mono, lr (Left-Right), ud (Up-Down), tc (Trinocular)
    # -dataType     : type of input imgs. 'erp' = Equirectangular projection, 'sphere' = s2 signal, 'all' = both type
    #########################################################################################################
    # Initialization
    super(Dataset3D60Fusion_2view, self).__init__()
    # Assertion
    assert curStage in splits
    assert (rootDir is not None) and (rootDir != '')
    assert pair in stereo_pairs

    assert view in ['Center_Left_Down/', 'Right/', 'Up/']
    # Member variable assignment
    self.rootDir = rootDir
    self.curStage = curStage
    self.height, self.width = shape
    self.pair = pair

    self.filenamesFile = filenamesFile
    self.baseline = 0.26  # left-right baseline
    self.maxDepth = maxDepth

    # rgb and gt depth dir
    self.prefix_l = os.path.join(self.rootDir, 'Center_Left_Down/')
    self.prefix_r = os.path.join(self.rootDir, 'Right/')
    self.prefix_u = os.path.join(self.rootDir, 'Up/')

    # input depth maps from stage 1
    # self.prefixPredDepth_l = os.path.join(self.inputDir, 'disp_pred2depth', 'Center_Left_Down/')
    # self.prefixPredDepth_r = os.path.join(self.inputDir, 'disp_pred2depth', 'Right/')
    # self.prefixPredDepth_u = os.path.join(self.inputDir, 'disp_pred2depth', 'Up/')
    # self.prefixConfMap_l = os.path.join(self.inputDir, 'conf_map', 'Center_Left_Down/')
    # self.prefixConfMap_r = os.path.join(self.inputDir, 'conf_map', 'Right/')
    # self.prefixConfMap_u = os.path.join(self.inputDir, 'conf_map', 'Up/')
    self.view = view
    self.prefixPredDepth = os.path.join(self.inputDir, self.view, 'disp_pred2depth')  # inpudir/Center_Left_Down/disp_pred2depth/Matterport3D/index_lr_l.disp_pred2depth.npz
    self.prefixConfMap = os.path.join(self.inputDir, self.view, 'conf_map')  # inpudir/Center_Left_Down/conf_map/Matterport3D/index_lr_l.conf_map.png

    self.processed = preprocess.get_transform_stage1(augment=False)  # transform of rgb images
    self.processed_depth = preprocess.get_transform_stage2()

    # self.cddt = CassiniDepthDispTransformer(height=self.height, width=self.width, maxDisp, maxDepth, baseline, device='cuda')

    # get file names
    self.fileNameList = self.__getFileList()
    self.phiMap = self.__genCassiniPhiMap()

    print("Dataset 3D60: Multi-views fish eye dataset. File list: {}. Num of files: {}. root dir: {}.".format(self.filenamesFile, len(self.fileNameList), self.rootDir))

  def __len__(self):
    return len(self.fileNameList)

  def __getFileList(self):
    fileNameList = []
    with open(self.filenamesFile) as f:
      lines = f.readlines()
      for line in lines:
        fileNameList.append(line.strip().split(" "))  # split by space
    return fileNameList

  def __getitem__(self, index):  #return data in disparity estimation task
    assert (self.fileNameList is not None) and (len(self.fileNameList) > 0)
    name = self.fileNameList[index]
    # left/down
    leftName = os.path.join(self.prefix_l, name[0][2:])
    leftDepthName = os.path.join(self.prefix_l, name[3][2:])

    # right
    rightName = os.path.join(self.prefix_r, name[1][2:])
    rightDepthName = os.path.join(self.prefix_r, name[4][2:])
    # up
    upName = os.path.join(self.prefix_u, name[2][2:])
    upDepthName = os.path.join(self.prefix_u, name[5][2:])

    if self.pair == 'lr':
      left = leftName
      right = rightName
      depth = leftDepthName
      depth_r = rightDepthName
      inputName = left.split('color')[0]
      pred_depth_1 = inputName.replace(self.prefix_l, self.prefixPredDepth) + 'lr_l' + '_disp_pred2depth.npz'
      pred_depth_2 = inputName.replace(self.prefix_r, self.prefixPredDepth_r) + 'lr_r' + '_disp_pred2depth.npz'
      conf_map_1 = inputName.replace(self.prefix_l, self.prefixConfMap_l) + 'lr_l' + '_conf_map.png'
      conf_map_2 = inputName.replace(self.prefix_r, self.prefixConfMap_r) + 'lr_r' + '_conf_map.png'
      rotate_vector = np.array([0, 0, 0]).astype(np.float32)
    elif self.pair == 'ud':
      left = upName
      right = leftName
      depth = upDepthName
      depth_r = leftDepthName
      inputName = left.split('color')[0]
      pred_depth_1 = inputName.replace(self.prefix_l, self.prefixPredDepth) + 'ud_u' + '_disp_pred2depth.npz'
      pred_depth_2 = inputName.replace(self.prefix_r, self.prefixPredDepth_r) + 'ud_d' + '_disp_pred2depth.npz'
      conf_map_1 = inputName.replace(self.prefix_l, self.prefixConfMap_l) + 'ud_u' + '_conf_map.png'
      conf_map_2 = inputName.replace(self.prefix_r, self.prefixConfMap_r) + 'ud_d' + '_conf_map.png'
      rotate_vector = np.array([0, 0, -np.pi / 2]).astype(np.float32)
    elif self.pair == 'ur':
      left = upName
      right = rightName
      depth = upDepthName
      depth_r = rightDepthName
      inputName = left.split('color')[0]
      pred_depth_1 = inputName.replace(self.prefix_l, self.prefixPredDepth) + 'ur_u' + '_disp_pred2depth.npz'
      pred_depth_2 = inputName.replace(self.prefix_r, self.prefixPredDepth_r) + 'ur_r' + '_disp_pred2depth.npz'
      conf_map_1 = inputName.replace(self.prefix_l, self.prefixConfMap_l) + 'ur_u' + '_conf_map.png'
      conf_map_2 = inputName.replace(self.prefix_r, self.prefixConfMap_r) + 'ur_r' + '_conf_map.png'
      rotate_vector = np.array([0, 0, -np.pi / 4]).astype(np.float32)

    R = cv2.Rodrigues(rotate_vector)[0]

    leftRGB = np.array(Image.open(left).convert('RGB'))
    rightRGB = np.array(Image.open(right).convert('RGB'))
    leftRGB = erp2rect_cassini(leftRGB, R, self.height, self.width, devcice='cpu').astype(np.uint8)
    rightRGB = erp2rect_cassini(rightRGB, R, self.height, self.width, devcice='cpu').astype(np.uint8)

    leftDepth = np.array(cv2.imread(depth, cv2.IMREAD_ANYDEPTH)).astype(np.float32)
    rightDepth = np.array(cv2.imread(depth_r, cv2.IMREAD_ANYDEPTH)).astype(np.float32)
    leftDepth = erp2rect_cassini(leftDepth, R, self.height, self.width, devcice='cpu')
    rightDepth = erp2rect_cassini(rightDepth, R, self.height, self.width, devcice='cpu')

    leftRGB = cv2.resize(leftRGB, (self.width, self.height))
    rightRGB = cv2.resize(rightRGB, (self.width, self.height))
    leftDepth = cv2.resize(leftDepth, (self.width, self.height))
    rightDepth = cv2.resize(rightDepth, (self.width, self.height))

    #conf maps
    confs = []
    conf = cv2.imread(conf_map_1)
    confs.append(np.expand_dims((conf[:, :, 0] / 255.0).astype(np.float32), axis=0))
    conf = cv2.imread(conf_map_2)
    confs.append(np.expand_dims((conf[:, :, 0] / 255.0).astype(np.float32), axis=0))

    # depths input
    depths = []
    inp = np.expand_dims(np.load(pred_depth_1)['arr_0'].astype(np.float32), axis=-1)
    depths.append(self.processed_depth(inp))
    inp = np.expand_dims(np.load(pred_depth_2)['arr_0'].astype(np.float32), axis=-1)
    depths.append(self.processed_depth(inp))

    leftImg, rightImg, depthMap = leftRGB, rightRGB, leftDepth

    depthMap[depthMap > self.maxDepth] = 0.0
    gt = np.squeeze(depthMap, axis=-1)
    gt = np.ascontiguousarray(gt, dtype=np.float32)

    leftRGB = leftRGB.astype(np.uint8)
    rightRGB = rightRGB.astype(np.uint8)

    # print(np.max(leftImg), np.min(leftImg))
    rgbs = []
    rgbs.append(self.processed(leftImg))
    rgbs.append(self.processed(rightImg))

    #leftDepth = torch.from_numpy(leftDepth).unsqueeze_(0)

    # self.gt[index], depthes, confs, rgbs, gt
    return depth, depths, confs, rgbs, gt  # gt name, input depth, confs, rgbs, gt

  def __genCassiniPhiMap(self):
    phi_l_start = 0.5 * np.pi - (0.5 * np.pi / self.width)
    phi_l_end = -0.5 * np.pi
    phi_l_step = np.pi / self.width
    phi_l_range = np.arange(phi_l_start, phi_l_end, -phi_l_step)
    phi_l_map = np.array([phi_l_range for j in range(self.height)]).astype(np.float32)
    return phi_l_map

  def __depth2disp(self, depthMap):
    mask_depth_0 = depthMap == 0
    invMask = (depthMap <= 0) | (depthMap > self.maxDepth)
    depth_not_0 = np.ma.array(depthMap, mask=invMask)
    phi_l_map = self.phiMap
    disp = self.width * (np.arcsin(
        np.clip(
            (depth_not_0 * np.sin(phi_l_map) + self.baseline) / np.sqrt(depth_not_0 * depth_not_0 + self.baseline * self.baseline - 2 * depth_not_0 * self.baseline * np.cos(phi_l_map + np.pi / 2)),
            -1,
            1)) - phi_l_map) / np.pi
    disp = disp.filled(np.nan)
    disp[disp < 0] = 0
    return disp

  def __rotateERP(self, erp, angle):
    w = erp.shape[-1]  # input erp image shape: c*h*w
    roll_idx = int(-angle / (2 * np.pi) * w)
    erp2 = np.roll(erp, roll_idx, -1)
    return erp2

  def __turnBack(self, leftRGB, rightRGB, rightDepth):
    angle = np.pi
    leftBack = self.__rotateERP(leftRGB, angle)
    rightBack = self.__rotateERP(rightRGB, angle)
    rightDepthBack = self.__rotateERP(rightDepth, angle)
    return rightBack, leftBack, rightDepthBack


class Dataset3D60Fusion_3view(Dataset):
  #360D Dataset#
  # fusion of all 3 views (lr_l,lr_r,ud_u,ud_d,ur_u,ur_r)
  def __init__(
      self,
      filenamesFile,
      rootDir='../../datasets/3D60/',  #rgb, disp,depth 
      inputDir='',  #depth from disparity stage
      curStage='training',
      shape=(512, 256),
      maxDepth=20.0,
      view='Center_Left_Down/'):  # 3D60 is a indoor dataset and set max depth as 20 meters
    #########################################################################################################
    # Arguments:
    # -filenamesFile: Absolute path to the aforementioned filenames .txt file
    # -transform    : (Optional) transform to be applied on a sample
    # -mode         : Dataset mode. Available options: mono, lr (Left-Right), ud (Up-Down), tc (Trinocular)
    # -dataType     : type of input imgs. 'erp' = Equirectangular projection, 'sphere' = s2 signal, 'all' = both type
    #########################################################################################################
    # Initialization
    super(Dataset3D60Fusion_3view, self).__init__()
    # Assertion
    assert curStage in splits
    assert (rootDir is not None) and (rootDir != '')

    assert view in ['Center_Left_Down/', 'Right/', 'Up/']
    # Member variable assignment
    self.rootDir = rootDir
    self.inputDir = inputDir
    self.curStage = curStage
    self.height, self.width = shape

    self.filenamesFile = filenamesFile
    self.baseline = 0.26  # left-right baseline
    self.maxDepth = maxDepth

    # rgb and gt depth dir
    self.prefix_l = os.path.join(self.rootDir, 'Center_Left_Down/')
    self.prefix_r = os.path.join(self.rootDir, 'Right/')
    self.prefix_u = os.path.join(self.rootDir, 'Up/')

    # input depth maps from stage 1
    # self.prefixPredDepth_l = os.path.join(self.inputDir, 'disp_pred2depth', 'Center_Left_Down/')
    # self.prefixPredDepth_r = os.path.join(self.inputDir, 'disp_pred2depth', 'Right/')
    # self.prefixPredDepth_u = os.path.join(self.inputDir, 'disp_pred2depth', 'Up/')
    # self.prefixConfMap_l = os.path.join(self.inputDir, 'conf_map', 'Center_Left_Down/')
    # self.prefixConfMap_r = os.path.join(self.inputDir, 'conf_map', 'Right/')
    # self.prefixConfMap_u = os.path.join(self.inputDir, 'conf_map', 'Up/')
    self.view = view
    self.prefixPredDepth = os.path.join(self.inputDir, self.view, 'disp_pred2depth/')  # inpudir/Center_Left_Down/disp_pred2depth/Matterport3D/index_lr_l.disp_pred2depth.npz
    self.prefixConfMap = os.path.join(self.inputDir, self.view, 'conf_map/')  # inpudir/Center_Left_Down/conf_map/Matterport3D/index_lr_l.conf_map.png

    self.processed = preprocess.get_transform_stage1(augment=False)  # transform of rgb images
    self.processed_depth = preprocess.get_transform_stage2()

    # self.cddt = CassiniDepthDispTransformer(height=self.height, width=self.width, maxDisp, maxDepth, baseline, device='cuda')

    # get file names
    self.fileNameList = self.__getFileList()
    self.phiMap = self.__genCassiniPhiMap()

    print("Dataset 3D60: Multi-views fish eye dataset. File list: {}. Num of files: {}. root dir: {}.".format(self.filenamesFile, len(self.fileNameList), self.rootDir))

  def __len__(self):
    return len(self.fileNameList)

  def __getFileList(self):
    fileNameList = []
    with open(self.filenamesFile) as f:
      lines = f.readlines()
      for line in lines:
        fileNameList.append(line.strip().split(" "))  # split by space
    return fileNameList

  def __getitem__(self, index):  #return data in disparity estimation task
    assert (self.fileNameList is not None) and (len(self.fileNameList) > 0)
    name = self.fileNameList[index]
    # left/down
    leftName = os.path.join(self.prefix_l, name[0][2:])
    leftDepthName = os.path.join(self.prefix_l, name[3][2:])

    # right
    rightName = os.path.join(self.prefix_r, name[1][2:])
    rightDepthName = os.path.join(self.prefix_r, name[4][2:])
    # up
    upName = os.path.join(self.prefix_u, name[2][2:])
    upDepthName = os.path.join(self.prefix_u, name[5][2:])

    left = leftName
    right = rightName
    up = upName
    depth = leftDepthName

    # RGB:
    rgbs = []
    rotate_vector = np.array([0, 0, 0]).astype(np.float32)
    R = cv2.Rodrigues(rotate_vector)[0]
    leftRGB = np.array(Image.open(left).convert('RGB'))
    rightRGB = np.array(Image.open(right).convert('RGB'))
    leftRGB = erp2rect_cassini(leftRGB, R, self.height, self.width, devcice='cpu').astype(np.uint8)
    rightRGB = erp2rect_cassini(rightRGB, R, self.height, self.width, devcice='cpu').astype(np.uint8)

    leftDepth_erp = np.array(cv2.imread(depth, cv2.IMREAD_ANYDEPTH)).astype(np.float32)
    leftDepth = erp2rect_cassini(leftDepth_erp, R, self.height, self.width, devcice='cpu')

    # rotate_vector = np.array([0, 0, -np.pi / 2]).astype(np.float32)
    # R = cv2.Rodrigues(rotate_vector)[0]
    upRGB = np.array(Image.open(up).convert('RGB'))
    upRGB = erp2rect_cassini(upRGB, R, self.height, self.width, devcice='cpu').astype(np.uint8)
    rgbs.append(self.processed(leftRGB))
    rgbs.append(self.processed(rightRGB))
    rgbs.append(self.processed(upRGB))

    # depth and confs
    depths = []
    confs = []
    inputName = left.split('color')[0]
    for id in ['lr_l', 'lr_r', 'ud_u', 'ud_d', 'ur_u', 'ur_r']:
      pred_depth_name = inputName.replace(self.prefix_l, self.prefixPredDepth) + id + '_disp_pred2depth.npz'
      conf_map_name = inputName.replace(self.prefix_l, self.prefixConfMap) + id + '_conf_map.png'
      inp = np.expand_dims(np.load(pred_depth_name)['arr_0'].astype(np.float32), axis=-1)
      depths.append(self.processed_depth(inp))
      conf = cv2.imread(conf_map_name)
      confs.append(np.expand_dims((conf[:, :, 0] / 255.0).astype(np.float32), axis=0))

    #leftDepth[leftDepth > self.maxDepth] = self.maxDepth  # 0.0
    # gt = np.squeeze(leftDepth, axis=-1)
    gt = np.ascontiguousarray(leftDepth, dtype=np.float32)

    #gt_erp = np.ascontiguousarray(leftDepth_erp, dtype=np.float32)

    leftRGB = leftRGB.astype(np.uint8)
    rightRGB = rightRGB.astype(np.uint8)

    #leftDepth = torch.from_numpy(leftDepth).unsqueeze_(0)
    return depth, depths, confs, rgbs, gt  # gt name, input depth, confs, rgbs, gt

  def __genCassiniPhiMap(self):
    phi_l_start = 0.5 * np.pi - (0.5 * np.pi / self.width)
    phi_l_end = -0.5 * np.pi
    phi_l_step = np.pi / self.width
    phi_l_range = np.arange(phi_l_start, phi_l_end, -phi_l_step)
    phi_l_map = np.array([phi_l_range for j in range(self.height)]).astype(np.float32)
    return phi_l_map

  def __depth2disp(self, depthMap):
    mask_depth_0 = depthMap == 0
    invMask = (depthMap <= 0) | (depthMap > self.maxDepth)
    depth_not_0 = np.ma.array(depthMap, mask=invMask)
    phi_l_map = self.phiMap
    disp = self.width * (np.arcsin(
        np.clip(
            (depth_not_0 * np.sin(phi_l_map) + self.baseline) / np.sqrt(depth_not_0 * depth_not_0 + self.baseline * self.baseline - 2 * depth_not_0 * self.baseline * np.cos(phi_l_map + np.pi / 2)),
            -1,
            1)) - phi_l_map) / np.pi
    disp = disp.filled(np.nan)
    disp[disp < 0] = 0
    return disp

  def __rotateERP(self, erp, angle):
    w = erp.shape[-1]  # input erp image shape: c*h*w
    roll_idx = int(-angle / (2 * np.pi) * w)
    erp2 = np.roll(erp, roll_idx, -1)
    return erp2


if __name__ == '__main__':
  from tqdm import tqdm
  os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # enable openexr
  # NOTE: test 3D60 disparity
  da = Dataset3D60Disparity(filenamesFile='./3d60_val.txt', rootDir='../../../datasets/3D60/', curStage='validation', shape=(512, 256), crop=False, pair='all', flip=False, maxDepth=20.0)
  myDL = torch.utils.data.DataLoader(da, batch_size=1, num_workers=1, pin_memory=False, shuffle=False)
  maxDisp = 0
  for id, batch in enumerate(tqdm(myDL, desc='Train iter')):
    if id < 10:
      if random.random() < 0.5:
        left = batch['leftImg']
        right = batch['rightImg']
        disp = batch['dispMap']
      else:
        left = batch['leftImg_flip']
        right = batch['rightImg_flip']
        disp = batch['dispMap_flip']
      print(disp.shape)
      print(torch.max(left), torch.min(left), torch.max(right), torch.min(right))
      disp[torch.isnan(disp)] = 0.0
      disp[(disp > 192)] = 0.0
      print(torch.max(disp), torch.min(disp))
      disp = (disp - torch.min(disp)) / (torch.max(disp) - torch.min(disp))
      left = (left - torch.min(left)) / (torch.max(left) - torch.min(left))
      right = (right - torch.min(right)) / (torch.max(right) - torch.min(right))
      torchvision.utils.save_image(left, 'ca_{}_l.png'.format(str(id)))
      torchvision.utils.save_image(right, 'ca_{}_r.png'.format(str(id)))
      torchvision.utils.save_image(disp, 'ca_{}_disp.png'.format(str(id)))
    # test output disparity
    maxDisp = max(maxDisp, torch.max(batch['dispMap']))
  print(maxDisp)

  # # NOTE: test 3D60 fusion
  # da = Dataset3D60Fusion_3view(filenamesFile='./3d60_test.txt', rootDir='../../../datasets/3D60/', inputDir='../outputs/pred_3D60/', curStage='testing')
  # myDL = torch.utils.data.DataLoader(da, batch_size=1, num_workers=1, pin_memory=False, shuffle=False)
  # for id, [depth_name, depths, confs, rgbs, gt] in enumerate(tqdm(myDL, desc='Train iter')):
  #   print(depth_name)
  #   print("depths: ", len(depths), depths[0].shape)
  #   print("confs: ", len(confs), confs[0].shape)
  #   print("rgbs: ", len(rgbs), rgbs[0].shape)
  #   print("gt: ", gt.shape, torch.max(gt), torch.min(gt))
  #   for i in range(len(depths)):
  #     d = (depths[i] - torch.min(depths[i])) / (torch.max(depths[i]) - torch.min(depths[i]))
  #     torchvision.utils.save_image(d, str(id) + '_depth_' + str(i) + '.png')
  #     torchvision.utils.save_image(confs[i], str(id) + '_conf_' + str(i) + '.png')
  #   for i in range(len(rgbs)):
  #     img = rgbs[i].squeeze().numpy().transpose((1, 2, 0))
  #     img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
  #     cv2.imwrite(str(id) + '_rgb_' + str(i) + '.png', img)
  #   if id > 5:
  #     break
