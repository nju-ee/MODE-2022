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
from utils.ERPandCassini import ERP2CA, CA2ERP
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
# inputCamPairs = ['12', '21', '12', '21', '12', '21']
inputCamPairs = ['12']
inputRGBImgIds = [0, 1]


class Dataset3D60(Dataset):
  #360D Dataset#
  def __init__(self,
               filenamesFile,
               rootDir='../../datasets/3D60/',
               interDir='outputs/3D60',
               mode='disparity',
               curStage='training',
               shape=(512,
                      256),
               crop=False,
               catEquiInfo=False,
               soiled=False,
               shuffleOrder=False,
               inputRGB=True,
               needMask=False,
               camPairs=inputCamPairs,
               rgbIds=inputRGBImgIds,
               copyFusion=True,
               maxDepth=20.0,
               saveOut=False):  # 3D60 is a indoor dataset and set max depth as 20 meters
    #########################################################################################################
    # Arguments:
    # -filenamesFile: Absolute path to the aforementioned filenames .txt file
    # -transform    : (Optional) transform to be applied on a sample
    # -mode         : Dataset mode. Available options: mono, lr (Left-Right), ud (Up-Down), tc (Trinocular)
    # -dataType     : type of input imgs. 'erp' = Equirectangular projection, 'sphere' = s2 signal, 'all' = both type
    #########################################################################################################
    # Initialization
    super(Dataset3D60, self).__init__()
    # Assertion
    assert mode in dataModes
    assert curStage in splits
    assert (rootDir is not None) and (rootDir != '')
    if mode == 'intermedia_fusion':
      assert (interDir is not None) and (interDir != '')
    assert (camPairs is not None) and (len(camPairs) > 0)
    if inputRGB:
      assert (rgbIds is not None) and (len(rgbIds) > 0)
    # Member variable assignment
    self.mode = mode
    self.rootDir = rootDir
    self.interDir = interDir
    self.curStage = curStage
    self.height, self.width = shape
    self.catEquiInfo = catEquiInfo
    self.crop = crop
    self.soiled = soiled
    self.shuffleOrder = shuffleOrder
    self.inputRGB = inputRGB
    self.needMask = needMask
    self.camPairs = camPairs
    self.filenamesFile = filenamesFile
    self.baseline = 0.26  # left-right baseline
    self.maxDepth = maxDepth

    self.copyFusion = copyFusion

    self.prefix_l = os.path.join(self.rootDir, 'Center_Left_Down/')
    self.prefix_r = os.path.join(self.rootDir, 'Right/')
    self.prefix_u = os.path.join(self.rootDir, 'Up/')

    self.prefixInter_l = os.path.join(self.interDir, 'Center_Left_Down/')
    self.prefixInter_r = os.path.join(self.interDir, 'Right/')
    self.prefixInter_u = os.path.join(self.interDir, 'Up/')

    self.e2ca = ERP2CA(self.width, self.height, self.height, self.width, False)
    self.ca2e = CA2ERP(self.width, self.height, self.height, self.width, False)

    self.processed = preprocess.get_transform(augment=False)  # transform of rgb images
    if self.inputRGB:
      self.rgbId = rgbIds  # use which pairs of RGB images
    # get file names
    self.fileNameList = self.__getFileList()
    self.phiMap = self.__genCassiniPhiMap()

    print("Dataset 3D60: Multi-views fish eye dataset. Num of files: {}".format(len(self.fileNameList)))
    print("root dir: {}. intermedia dir: {}.".format(self.rootDir, self.interDir))
    print("Camera pairs: {}.".format(self.camPairs))
    if self.mode == 'disparity':
      #self.dispList = self.__getDisparityList()
      print("Dataset in mode [ {} ], at stage [ {} ], num of items: {}".format(self.mode, self.curStage, len(self.fileNameList)))
    elif self.mode == 'fusion':
      #self.fusionList = self.__getFusionList()
      print("Dataset in mode [ {} ], at stage [ {} ], num of items: {}".format(self.mode, self.curStage, len(self.fileNameList)))
    elif self.mode == 'intermediate':
      #self.interFusionList = self.__getIntermediateList()
      print("Dataset in mode [ {} ], at stage [ {} ], num of items: {}".format(self.mode, self.curStage, len(self.fileNameList)))
    else:
      raise TypeError("dataset mode error: {} is not a valid mode".format(self.mode))

  def __getitem__(self, index):
    inputs = {}
    if self.mode == 'disparity':
      inputs = self.__getDisparityData(index)
    elif self.mode == 'fusion':
      inputs = self.__getFusionData(index)
    elif self.mode == 'intermedia_fusion':
      inputs = self.__getIntermediateData(index)
    else:
      raise TypeError("dataset mode error: {} is not a valid mode".format(self.mode))
    return inputs

  def __len__(self):
    return len(self.fileNameList)

  def __getFileList(self):
    fileNameList = []
    with open(self.filenamesFile) as f:
      lines = f.readlines()
      for line in lines:
        fileNameList.append(line.strip().split(" "))  # split by space
    return fileNameList

  def __getDisparityList(self):  # return data names as a list in disparity estimation task
    dispList = []
    for fn in self.fileNameList:
      numsuf = fn[1].split('/')[-1]
      if self.soiled:
        dRgb = os.path.join(self.rootDir, fn[0], 'rgb_soiled')
      else:
        dRgb = os.path.join(self.rootDir, fn[0], 'rgb')
      dDisp = os.path.join(self.rootDir, fn[0], 'disp')
      dDepth = os.path.join(self.rootDir, fn[0], 'depth')
      depthName = os.path.join(dDepth, numsuf + '_depth.npy')
      for cp in self.camPairs:
        l = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[0] + '.png')
        r = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[1] + '.png')
        disp = os.path.join(dDisp, numsuf + '_' + cp + '_disp.npy')
        dispList.append([l, r, disp, cp])
    return dispList

  def __getFusionList(self):  # return data names as a list in depth fusion (full stage) task
    fusionList = []
    for fn in self.fileNameList:
      numsuf = fn[1].split('/')[-1]
      if self.soiled:
        dRgb = os.path.join(self.rootDir, fn[0], 'rgb_soiled')
      else:
        dRgb = os.path.join(self.rootDir, fn[0], 'rgb')
      dDisp = os.path.join(self.rootDir, fn[0], 'disp')
      dDepth = os.path.join(self.rootDir, fn[0], 'depth')
      depthName = os.path.join(dDepth, numsuf + '_depth.npy')
      fusionL = []
      fusionR = []
      for cp in self.camPairs:
        l = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[0] + '.png')
        r = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[1] + '.png')
        disp = os.path.join(dDisp, numsuf + '_' + cp + '_disp.npy')
        fusionL.append(l)
        fusionR.append(r)
      fusionList.append([fusionL.copy(), fusionR.copy(), depthName])
      if self.curStage == 'training' and self.needMask:
        maskName = os.path.join(self.rootDir, fn[0], 'soil_mask', numsuf + '_soil_mask.npy')
        fusionList[-1].append(maskName)
    return fusionList

  def __getIntermediateList(self):  # return data names as a list in intermediate fusion (from predict depth maps) task
    fusionList = []
    for fn in self.fileNameList:
      numsuf = fn[1].split('/')[-1]
      if self.soiled:
        dRgb = os.path.join(self.rootDir, fn[0], 'rgb_soiled')
      else:
        dRgb = os.path.join(self.rootDir, fn[0], 'rgb')
      if self.soiled:
        dInput = os.path.join(self.interDir, fn[0], 'pred_depth_soiled')
      else:
        dInput = os.path.join(self.interDir, fn[0], 'pred_depth')
      dGt = os.path.join(self.rootDir, fn[0], 'depth')
      depthName = os.path.join(dGt, numsuf + '_depth.npy')
      depthOriName = os.path.join(self.depthOriDir, '0' + numsuf + '.tiff')  # tiff format original invDepth file
      fusion = []
      rgb = []
      for cp in self.camPairs:
        l = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[0] + '.png')
        r = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[1] + '.png')
        rgb.append([l, r])
        if self.soiled:
          fusion.append(os.path.join(dInput, fn[1] + '_' + cp + '_pred_depth_soiled.npy'))
        else:
          fusion.append(os.path.join(dInput, fn[1] + '_' + cp + '_pred_depth.npy'))
      fusionList.append([fusion.copy(), depthName, rgb.copy(), depthOriName])
      if self.curStage == 'training' and self.needMask:
        maskName = os.path.join(self.rootDir, fn[0], 'soil_mask', numsuf + '_soil_mask.npy')
        fusionList[-1].append(maskName)
    return fusionList

  def __getDispMask(self):
    masks = []
    for cp in self.camPairs:
      lname = os.path.join(self.rootDir, self.maskCassiniName + '_' + cp + '_' + cp[0] + '.png')
      rname = os.path.join(self.rootDir, self.maskCassiniName + '_' + cp + '_' + cp[1] + '.png')
      lmask = cv2.imread(lname, cv2.IMREAD_GRAYSCALE).astype(np.float32)
      rmask = cv2.imread(rname, cv2.IMREAD_GRAYSCALE).astype(np.float32)
      lmask = lmask / 255.0
      rmask = rmask / 255.0
      overLap = lmask + rmask
      invalid = (overLap > 0)
      print(invalid.shape)
      m = [lmask, rmask, invalid]
      # invalidSave = invalid * 255
      # invalidSave = invalidSave.astype(np.uint8)
      # cv2.imwrite('invalidMask_' + cp + '.png', invalidSave)
      masks.append(m.copy())
    return masks

  def __getDisparityData(self, index):  #return data in disparity estimation task
    assert (self.fileNameList is not None) and (len(self.fileNameList) > 0)
    name = self.fileNameList[index]
    leftName = os.path.join(self.prefix_l, name[0][2:])
    leftDepthName = os.path.join(self.prefix_l, name[3][2:])
    rightName = os.path.join(self.prefix_r, name[1][2:])
    rightDepthName = os.path.join(self.prefix_r, name[4][2:])
    leftRGB = np.array(Image.open(leftName).convert('RGB'))
    rightRGB = np.array(Image.open(rightName).convert('RGB'))
    leftRGB = cv2.resize(leftRGB, (self.height, self.width))
    rightRGB = cv2.resize(rightRGB, (self.height, self.width))

    leftRGB = leftRGB.transpose((2, 0, 1)).astype(np.float32)
    rightRGB = rightRGB.transpose((2, 0, 1)).astype(np.float32)
    leftDepth = np.array(cv2.imread(leftDepthName, cv2.IMREAD_ANYDEPTH))
    rightDepth = np.array(cv2.imread(rightDepthName, cv2.IMREAD_ANYDEPTH))
    leftDepth = cv2.resize(leftDepth, (self.height, self.width))
    rightDepth = cv2.resize(rightDepth, (self.height, self.width))

    tb = self.curStage == 'training' and random.random() > 0.5
    if tb:
      leftImg, rightImg, depth = self.__turnBack(leftRGB, rightRGB, rightDepth)
    else:
      leftImg, rightImg, depth = leftRGB, rightRGB, leftDepth

    leftImg = torch.from_numpy(leftImg).unsqueeze_(0)
    rightImg = torch.from_numpy(rightImg).unsqueeze_(0)
    depthMap = torch.from_numpy(depth).unsqueeze_(0).unsqueeze_(0)
    leftImg, rightImg = self.e2ca.transPairs(leftImg, rightImg, '0')
    depthMap = self.e2ca.trans(depthMap, '0')
    leftImg = leftImg.squeeze_(0).numpy().transpose((1, 2, 0))
    rightImg = rightImg.squeeze_(0).numpy().transpose((1, 2, 0))
    depthMap = depthMap.squeeze_(0).squeeze_(0).numpy()
    depthMap[depthMap > self.maxDepth] = 0.0

    # leftRGB[leftRGB > 255] = 255
    # leftRGB[leftRGB < 0] = 0
    # rightRGB[rightRGB > 255] = 255
    # rightRGB[rightRGB < 0] = 0
    # leftRGB = leftRGB.astype(np.uint8)
    # rightRGB = rightRGB.astype(np.uint8)

    # print(np.max(leftImg), np.min(leftImg))

    dispMap = self.__depth2disp(depthMap)

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

      if self.catEquiInfo:
        eq = self.equi_info[:, y1:y1 + th, x1:x1 + tw]
        leftImg = torch.cat([leftImg, eq], dim=0)
        rightImg = torch.cat([rightImg, eq], dim=0)
      dispMap = torch.from_numpy(dispMap).unsqueeze_(0)
      data = {'leftImg': leftImg, 'rightImg': rightImg, 'dispMap': dispMap, 'leftNames': leftName}
    else:
      leftImg = self.processed(leftImg)
      rightImg = self.processed(rightImg)
      if self.catEquiInfo:
        eq = self.equi_info
        leftImg = torch.cat([leftImg, eq], dim=0)
        rightImg = torch.cat([rightImg, eq], dim=0)
      dispMap = torch.from_numpy(dispMap).unsqueeze_(0)
      data = {'leftImg': leftImg, 'rightImg': rightImg, 'dispMap': dispMap, 'leftNames': leftName}
    return data

  def __getFusionData(self, index):  # return data in depth fusion (full stage) task
    assert (self.fileNameList is not None) and (len(self.fileNameList) > 0)
    name = self.fileNameList[index]
    leftName = os.path.join(self.prefix_l, name[0][2:])
    leftDepthName = os.path.join(self.prefix_l, name[3][2:])
    rightName = os.path.join(self.prefix_r, name[1][2:])
    rightDepthName = os.path.join(self.prefix_r, name[4][2:])
    leftRGB = np.array(Image.open(leftName).convert('RGB'))
    rightRGB = np.array(Image.open(rightName).convert('RGB'))
    leftRGB = leftRGB.transpose((2, 0, 1)).astype(np.float32)
    rightRGB = rightRGB.transpose((2, 0, 1)).astype(np.float32)
    leftDepth = np.array(cv2.imread(leftDepthName, cv2.IMREAD_ANYDEPTH))
    rightDepth = np.array(cv2.imread(rightDepthName, cv2.IMREAD_ANYDEPTH))

    leftRgbBack, rightRgbBack, depthBack = self.__turnBack(leftRGB, rightRGB, rightDepth)

    leftRGB = torch.from_numpy(leftRGB).unsqueeze_(0)
    rightRGB = torch.from_numpy(rightRGB).unsqueeze_(0)
    leftRgbBack = torch.from_numpy(leftRgbBack).unsqueeze_(0)
    rightRgbBack = torch.from_numpy(rightRgbBack).unsqueeze_(0)
    depthMap = torch.from_numpy(leftDepth).unsqueeze_(0).unsqueeze_(0)

    leftRGB, rightRGB = self.e2ca.transPairs(leftRGB, rightRGB, '0')
    leftRgbBack, rightRgbBack = self.e2ca.transPairs(leftRgbBack, rightRgbBack, '0')
    depthMap = self.e2ca.trans(depthMap, '0')
    leftRGB = leftRGB.squeeze_(0).numpy().transpose((1, 2, 0))
    rightRGB = rightRGB.squeeze_(0).numpy().transpose((1, 2, 0))
    leftRgbBack = leftRgbBack.squeeze_(0).numpy().transpose((1, 2, 0))
    rightRgbBack = rightRgbBack.squeeze_(0).numpy().transpose((1, 2, 0))
    depthMap = depthMap.squeeze_(0)

    imgPairs = []
    leftNames = []
    numDepth = len(self.camPairs) if self.copyFusion else 2
    for i in range(numDepth):
      if (self.camPairs[i] == '12'):
        leftImg = self.processed(leftRGB)
        rightImg = self.processed(rightRGB)
        leftNames.append(leftName)
      else:
        leftImg = self.processed(leftRgbBack)
        rightImg = self.processed(rightRgbBack)
        leftNames.append(rightName)
      if self.catEquiInfo:
        eq = self.equi_info
        leftImg = torch.cat([leftImg, eq], dim=0)
        rightImg = torch.cat([rightImg, eq], dim=0)
      imgPairs.append([leftImg, rightImg])
    rgbImgs = []
    if self.inputRGB:
      leftImg = leftRGB.copy()
      rightImg = rightRGB.copy()
      leftBackImg = leftRgbBack.copy()
      rightBackImg = rightRgbBack.copy()
      rgbImgs.append(self.processed(leftImg))
      rgbImgs.append(self.processed(rightImg))
      rgbImgs.append(self.processed(leftBackImg))
      rgbImgs.append(self.processed(rightBackImg))
      rgbImgs = torch.cat(rgbImgs, dim=0)
    #depthMap = torch.from_numpy(depthMap).unsqueeze_(0)
    detphMapERP = torch.from_numpy(leftDepth).unsqueeze_(0)
    data = {'imgPairs': imgPairs, 'depthMap': depthMap, 'leftNames': leftNames, 'depthMapERP': detphMapERP}
    if self.inputRGB:
      data['rgbImgs'] = rgbImgs
    if self.curStage == 'training' and self.needMask:
      assert len(name) == 4
      maskName = name[3]
      mask = np.load(maskName).astype(np.float32)
      mask = torch.from_numpy(mask).unsqueeze_(0)
      data['soilMask'] = mask
    return data

  def __getIntermediateData(self, index):  # return data in intermediate fusion (from predict depth maps) task
    assert (self.fileNameList is not None) and (len(self.fileNameList) > 0)
    name = self.fileNameList[index]
    leftName = os.path.join(self.prefix_l, name[0][2:])
    leftDepthName = os.path.join(self.prefix_r, name[3][2:])
    rightName = os.path.join(self.prefix_l, name[1][2:])
    rightDepthName = os.path.join(self.prefix_r, name[4][2:])
    leftRGB = np.array(Image.open(leftName).convert('RGB'))
    rightRGB = np.array(Image.open(rightName).convert('RGB'))
    leftDepth = np.array(cv2.imread(leftDepthName, cv2.IMREAD_ANYDEPTH))
    rightDepth = np.array(cv2.imread(rightDepthName, cv2.IMREAD_ANYDEPTH))

    leftRgbBack, rightRgbBack, depthBack = self.__turnBack(leftRGB, rightRGB, rightDepth)
    leftRGB, rightRGB = self.e2ca.transPairs(leftRGB, rightRGB, '0')
    leftRgbBack, rightRgbBack = self.e2ca.transPairs(leftRgbBack, rightRgbBack, '0')
    depthMap = self.e2ca.trans(leftDepth, '0')

    namedepthLeft = name[0][1:][:-4] + '.npy'
    namedepthRight = name[3][1:][:-4] + '.npy'
    namedepthLeft = namedepthLeft.replace('color', 'depth_inter')
    namedepthRight = namedepthRight.replace('color', 'depth_inter')
    namedepthLeft = os.path.join(self.prefixInter_l, namedepthLeft)
    namedepthRight = os.path.join(self.prefixInter_r, namedepthRight)

    depthFusionLeft = np.load(namedepthLeft).astype(np.float32)
    depthFusionRight = np.load(namedepthRight).astype(np.float32)
    depthFusionLeft = torch.from_numpy(depthFusionLeft).unsqueeze_(0)
    depthFusionRight = torch.from_numpy(depthFusionRight).unsqueeze_(0)

    inputDepthMaps = []
    rgbImgs = []
    numDepth = len(self.camPairs) if self.copyFusion else 2
    r = list(range(numDepth))
    if self.shuffleOrder:
      random.shuffle(r)
    for i in r:
      if (self.camPairs[i] == '12'):
        inputDepthMaps.append(depthFusionLeft.clone())
      else:
        inputDepthMaps.append(depthFusionRight.clone())
    if self.inputRGB:
      leftImg = leftRGB.clone()
      rightImg = rightRGB.clone()
      leftBackImg = leftRgbBack.clone()
      rightBackImg = rightRgbBack.clone()
      rgbImgs.append(self.processed(leftImg))
      rgbImgs.append(self.processed(rightImg))
      rgbImgs.append(self.processed(leftBackImg))
      rgbImgs.append(self.processed(rightBackImg))
    rgbImgs = torch.cat(rgbImgs, dim=0)
    inputDepthMaps = torch.cat(inputDepthMaps, dim=0)

    depthMap = torch.from_numpy(depthMap).unsqueeze_(0)
    detphMapERP = torch.from_numpy(leftDepth).unsqueeze_(0)

    data = {'inputs': inputDepthMaps, 'depthMap': depthMap, 'leftNames': name[0], 'depthName': name[1], 'depthMapERP': detphMapERP}  # return index as well
    if self.inputRGB:
      data['rgbImgs'] = rgbImgs
    if self.curStage == 'training' and self.needMask:
      assert len(name) == 5
      maskName = name[4]
      mask = np.load(maskName).astype(np.float32)
      mask = torch.from_numpy(mask).unsqueeze_(0)
      data['soilMask'] = mask
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


if __name__ == '__main__':
  from tqdm import tqdm
  # for cur in splits:
  #   da = Multi360FusionSoiledDataset(rootDir='../../../datasets/Deep360/depth_on_1', curStage=cur)

  # da = Multi360FusionSoiledDataset(rootInputDir='../outputs/depth_on_1_inter', rootDepthDir='../../../datasets/Deep360/depth_on_1', curStage='training')
  # trainFusionDataLoader = torch.utils.data.DataLoader(da, batch_size=1, num_workers=1, pin_memory=False, shuffle=True)
  # for id, batch in enumerate(trainFusionDataLoader):
  #   # imagePairs = batch['inputs']
  #   # depth = batch['depthMap']
  #   # print(batch['leftName'])
  #   print(batch['soilMask'].shape)
  #   print(torch.max(batch['soilMask']), torch.min(batch['soilMask']))
  da = Dataset3D60(filenamesFile='./3d60_train.txt',
                   rootDir='../../../datasets/3D60/',
                   interDir='../outputs/OmniHouse_inter',
                   mode='disparity',
                   curStage='training',
                   shape=(1024,
                          512),
                   crop=True,
                   catEquiInfo=False,
                   soiled=False,
                   shuffleOrder=False,
                   inputRGB=True,
                   needMask=False)
  myDL = torch.utils.data.DataLoader(da, batch_size=6, num_workers=1, pin_memory=False, shuffle=False)
  maxDisp = 0
  for id, batch in enumerate(tqdm(myDL, desc='Train iter')):
    if id < 5:
      left = batch['leftImg']
      right = batch['rightImg']
      disp = batch['dispMap']
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
    # print(batch['leftNames'])
    # print(batch['leftImg'].shape)
    # print(batch['rightImg'].shape)
    # print(batch['dispMap'].shape)

    # test output fusion
    # print(len(batch['imgPairs']))
    # print(batch['leftNames'])
    # print(batch['imgPairs'][0][0].shape)
    # print(batch['depthMap'].shape)
    # print(batch['rgbImgs'].shape)
    # print(batch['soilMask'].shape)

    # test output inter fusion
    # print(batch['leftImg'].shape)
    # print(batch['rightImg'].shape)
    # print(batch['leftNames'])
    #print(torch.max(batch['dispMap']), torch.min(batch['dispMap']))
    # if (torch.max(batch['dispMap']) > 192):
    print(batch['leftNames'])
    maxDisp = max(maxDisp, torch.max(batch['dispMap']))
  print(maxDisp)
