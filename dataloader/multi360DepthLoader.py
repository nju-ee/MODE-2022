import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import random
from PIL import Image
import numpy as np

if __name__ == '__main__':
  import preprocess
else:
  from . import preprocess
"""
NOTE:

Multi view 360 depth Dataset
3000 scene points
each point contains 6 pairs of left-right cassini preojection images
(12,13,14,23,24,34), 6 disparity maps and 1 depth maps on view 1

Dataset directory structure:
Deep360
 - depth_on_1
   - ep1_500frames
   - ep2_500frames
   - ep3_500frames
   - ep4_500frames
   - ep5_500frames
   - ep6_500frames
     - training
     - testing
     - validation
       - rgb
       - disp
       - depth
       - rgb_soiled

"""

scenes = ['ep1_500frames', 'ep2_500frames', 'ep3_500frames', 'ep4_500frames', 'ep5_500frames', 'ep6_500frames']
splits = ['training', 'testing', 'validation']
dataType = ['rgb', 'disp', 'depth']
soiledType = ['glare', 'mud', 'water']
soiledNum = ['1_soiled_cam', '2_soiled_cam']
spotNum = ['2_spot', '3_spot', '4_spot', '5_spot', '6_spot']
percent = ['05percent', '10percent', '15percent', '20percent']

dataModes = ['disparity', 'fusion', 'intermedia_fusion']
inputCamPairs = ['12', '13', '14', '23', '24', '34']
inputRGBImgIds = [0, 5]


# TODO: test this dataset and fix bugs
# try to build one Dataset Class to implement reading data of disparity, fusion, intermedia_fusion, soiled data
class MultiViewDeep360Dataset(Dataset):
  def __init__(self,
               rootDir='../../datasets/Deep360/depth_on_1',
               interDir='outputs/depth_on_1_inter',
               mode='disparity',
               curStage='training',
               shape=(1024,
                      512),
               crop=False,
               catEquiInfo=False,
               soiled=False,
               shuffleOrder=False,
               inputRGB=True,
               needMask=False,
               camPairs=inputCamPairs,
               rgbIds=inputRGBImgIds):
    # NOTE：Arguments
    # rootDir: root directory of datasets (RGB, disparity, depth ground truth et.al)
    # interDir: root directory of intermediate results (predicted depth maps from the disparity model)
    # mode: usage and type of loaded data, must be one in dataModes
    # curStage: current working stage, training/validation/testing
    # shape: img shape of input data. a tuple of (height, width)
    # crop: Bool. if set True, will use random crop
    # catEquiInfo: Bool. if set True, will concat equi info as the 4th channel of rgb images
    # soiled: Bool. if set True, will use soiled data
    # shuffleOrder: Bool. if set True, will randomly shuffle the order of input depth maps
    # inputRGB: Bool. if set True, will return specific rgb images in fusion task
    # needMask: Bool. if set True, will return soiled mask when current is training

    # Initialization
    super(MultiViewDeep360Dataset, self).__init__()
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
    self.processed = preprocess.get_transform(augment=False)  # transform of rgb images
    if self.inputRGB:
      self.rgbId = rgbIds  # use which pairs of RGB images
    # get file names
    self.fileNameList = self.__getFileList()
    print("Dataset Deep360: Multi-views 360 depth. Num of files: {}".format(len(self.fileNameList)))
    print("Camera pairs: {}. RGB image ids: {}".format(self.camPairs, self.rgbId))
    self.dispList = []
    self.fusionList = []
    self.interFusionList = []
    if self.mode == 'disparity':
      self.dispList = self.__getDisparityList()
      print("Dataset in mode [ {} ], at stage [ {} ], num of items: {}".format(self.mode, self.curStage, len(self.dispList)))
    elif self.mode == 'fusion':
      self.fusionList = self.__getFusionList()
      print("Dataset in mode [ {} ], at stage [ {} ], num of items: {}".format(self.mode, self.curStage, len(self.fusionList)))
    elif self.mode == 'intermedia_fusion':
      self.interFusionList = self.__getIntermediateList()
      print("Dataset in mode [ {} ], at stage [ {} ], num of items: {}".format(self.mode, self.curStage, len(self.interFusionList)))
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
    if self.mode == 'disparity':
      return len(self.dispList)
    elif self.mode == 'fusion':
      return len(self.fusionList)
    elif self.mode == 'intermedia_fusion':
      return len(self.interFusionList)
    else:
      raise TypeError("dataset mode error: {} is not a valid mode".format(self.mode))

  def __getFileList(self):
    fileNameList = []
    if self.soiled:  # for soiled data
      if self.curStage == 'testing':
        for sub in scenes:
          for st in soiledType:
            for sn in soiledNum:
              for spot in spotNum:
                for p in percent:
                  names = os.listdir(os.path.join(self.rootDir, sub, self.curStage, 'rgb_soiled', st, sn, spot, p))
                  files = set()
                  for nn in names:
                    files.add(os.path.join(st, sn, spot, p, nn.split('_')[0]))
                  for nn in files:
                    fileNameList.append([os.path.join(sub, self.curStage), nn])
      else:
        for sub in scenes:
          gtDir = os.path.join(self.rootDir, sub, self.curStage, 'depth')
          fileNames = sorted(os.listdir(gtDir))
          for fn in fileNames:
            fileNameList.append([os.path.join(sub, self.curStage), fn.split('_')[0]])
    else:  # for normal data
      for sub in scenes:
        gtDir = os.path.join(self.rootDir, sub, self.curStage, 'depth')
        fileNames = sorted(os.listdir(gtDir))
        for fn in fileNames:
          fileNameList.append([os.path.join(sub, self.curStage), fn.split('_')[0]])
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
      fusionList.append([fusion.copy(), depthName, rgb.copy()])
      if self.curStage == 'training' and self.needMask:
        maskName = os.path.join(self.rootDir, fn[0], 'soil_mask', numsuf + '_soil_mask.npy')
        fusionList[-1].append(maskName)
    return fusionList

  def __getDisparityData(self, index):  #return data in disparity estimation task
    assert (self.dispList is not None) and (len(self.dispList) > 0)
    name = self.dispList[index]
    leftName = name[0]
    rightName = name[1]
    dispName = name[2]
    leftImg = Image.open(leftName).convert('RGB')
    rightImg = Image.open(rightName).convert('RGB')
    dispMap = np.load(dispName).astype(np.float32)
    dispMap = np.ascontiguousarray(dispMap, dtype=np.float32)
    cp = name[3]
    if self.crop:
      w, h = leftImg.size
      th, tw = 512, 256

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
    name = self.fusionList[index]
    imgPairs = []
    leftNames = []
    for i in range(len(self.camPairs)):
      leftName = name[0][i]
      leftNames.append(leftName)
      rightName = name[1][i]
      leftImg = Image.open(leftName).convert('RGB')
      rightImg = Image.open(rightName).convert('RGB')
      leftImg = self.processed(leftImg)
      rightImg = self.processed(rightImg)
      if self.catEquiInfo:
        eq = self.equi_info
        leftImg = torch.cat([leftImg, eq], dim=0)
        rightImg = torch.cat([rightImg, eq], dim=0)
      imgPairs.append([leftImg, rightImg])
    rgbImgs = []
    if self.inputRGB:
      for i in self.rgbId:
        leftImg = imgPairs[i][0].clone()
        rightImg = imgPairs[i][1].clone()
        rgbImgs.append(leftImg)
        rgbImgs.append(rightImg)
      rgbImgs = torch.cat(rgbImgs, dim=0)
    depthName = name[2]
    depthMap = np.load(depthName).astype(np.float32)
    depthMap = torch.from_numpy(depthMap).unsqueeze_(0)
    data = {'imgPairs': imgPairs, 'depthMap': depthMap, 'leftNames': leftNames}
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
    name = self.interFusionList[index]
    inputDepthMaps = []
    rgbImgs = []
    r = list(range(len(self.camPairs)))
    if self.shuffleOrder:
      random.shuffle(r)
    for i in r:
      inputName = name[0][i]
      inputD = np.load(inputName).astype(np.float32)
      inputD = torch.from_numpy(inputD).unsqueeze_(0)
      inputDepthMaps.append(inputD)
    if self.inputRGB:
      for i in self.rgbId:
        self.processed = preprocess.get_transform(augment=False)
        leftImg = Image.open(name[2][i][0]).convert('RGB')
        rightImg = Image.open(name[2][i][1]).convert('RGB')
        rgbImgs.append(self.processed(leftImg))
        rgbImgs.append(self.processed(rightImg))
      rgbImgs = torch.cat(rgbImgs, dim=0)
    inputDepthMaps = torch.cat(inputDepthMaps, dim=0)
    depthName = name[1]
    depthMap = np.load(depthName).astype(np.float32)
    depthMap = torch.from_numpy(depthMap).unsqueeze_(0)
    data = {'inputs': inputDepthMaps, 'depthMap': depthMap, 'leftNames': name[0], 'depthName': name[1]}
    if self.inputRGB:
      data['rgbImgs'] = rgbImgs
    if self.curStage == 'training' and self.needMask:
      assert len(name) == 4
      maskName = name[3]
      mask = np.load(maskName).astype(np.float32)
      mask = torch.from_numpy(mask).unsqueeze_(0)
      data['soilMask'] = mask
    return data


# TODO: build a dataset for 360D (binocular indoor omnidirectional dataset):
class Bi3D60Dataset(Dataset):
  def __init__(self):
    pass

  def __getitem__(self, index):
    pass

  def __len__(self):
    pass


# Dataloader disparity & fusion
class Multi360DepthDataset(Dataset):
  def __init__(self, dataUsage='disparity', crop=False, rootDir='../../datasets/Deep360/depth_on_1', curStage='training', shape=(1024, 512), catEquiInfo=False):
    assert dataUsage in ['disparity', 'fusion']
    assert curStage in splits
    self.dataUsage = dataUsage
    self.crop = crop
    self.rootDir = rootDir
    self.curStage = curStage
    self.height, self.width = shape
    self.catEquiInfo = catEquiInfo
    self.camPairs = ['12', '13', '14', '23', '24', '34']
    self.fileNameList, self.dispList, self.fusionList = self.__getFileList()  #file name list
    #print(self.fileNameList)
    print("current stage: {}, num of frames: {}, num of disp: {}, num of fusion: {}".format(self.curStage, len(self.fileNameList), len(self.dispList), len(self.fusionList)))
    if len(self.fileNameList) == 0 or len(self.dispList) == 0 or len(self.fusionList) == 0:
      raise ValueError("Error in initializing dataset: Empty File list!")
    # element in self.fileNameList: [root, name prefix]
    # element in self.dispList: [left name, right name, disp name]
    # element in self.fusionList: [[left names of cam pairs],[right names of cam pairs],depth name]
    self.processed = preprocess.get_transform(augment=False)
    if self.catEquiInfo:
      self.equi_info = preprocess.createAngleInfo(self.height, self.width, 'X')

  def __getitem__(self, index):
    if self.dataUsage == 'disparity':
      name = self.dispList[index]
      leftName = name[0]
      rightName = name[1]
      dispName = name[2]
      leftImg = Image.open(leftName).convert('RGB')
      rightImg = Image.open(rightName).convert('RGB')
      dispMap = np.load(dispName).astype(np.float32)
      dispMap = np.ascontiguousarray(dispMap, dtype=np.float32)
      cp = name[-2:]
      if self.crop:
        w, h = leftImg.size
        #th, tw = 512, 256
        th, tw = 576, 288  # for aanet

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
    else:
      name = self.fusionList[index]
      imgPairs = []
      leftNames = []
      for i in range(len(self.camPairs)):
        leftName = name[0][i]
        leftNames.append(leftName)
        rightName = name[1][i]
        leftImg = Image.open(leftName).convert('RGB')
        rightImg = Image.open(rightName).convert('RGB')
        leftImg = self.processed(leftImg)
        rightImg = self.processed(rightImg)
        if self.catEquiInfo:
          eq = self.equi_info
          leftImg = torch.cat([leftImg, eq], dim=0)
          rightImg = torch.cat([rightImg, eq], dim=0)
        imgPairs.append([leftImg, rightImg])
      depthName = name[2]
      depthMap = np.load(depthName).astype(np.float32)
      depthMap = torch.from_numpy(depthMap).unsqueeze_(0)
      data = {'imgPairs': imgPairs, 'depthMap': depthMap, 'leftNames': leftNames}
    return data

  def __len__(self):
    if self.dataUsage == 'disparity':
      return len(self.dispList)
    else:
      return len(self.fusionList)

  def __getFileList(self):
    fileNameList = []
    #subsets = os.listdir(self.rootDir)
    for sub in scenes:
      gtDir = os.path.join(self.rootDir, sub, self.curStage, 'depth')
      fileNames = sorted(os.listdir(gtDir))
      for fn in fileNames:
        fileNameList.append([os.path.join(sub, self.curStage), fn.split('_')[0]])
    dispList = []
    fusionList = []
    for fn in fileNameList:
      dRgb = os.path.join(self.rootDir, fn[0], 'rgb')
      dDisp = os.path.join(self.rootDir, fn[0], 'disp')
      dDepth = os.path.join(self.rootDir, fn[0], 'depth')
      fusionL = []
      fusionR = []
      depthName = os.path.join(dDepth, fn[1] + '_depth.npy')
      for cp in self.camPairs:
        l = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[0] + '.png')
        r = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[1] + '.png')
        disp = os.path.join(dDisp, fn[1] + '_' + cp + '_disp.npy')
        dispList.append([l, r, disp])
        fusionL.append(l)
        fusionR.append(r)
      fusionList.append([fusionL.copy(), fusionR.copy(), depthName])
    return fileNameList, dispList, fusionList


# Dataloader Intermedia fusion
class Multi360FusionDataset(Dataset):
  def __init__(self, rootInputDir='../../datasets/Deep360/depth_on_1', rootDepthDir='../../datasets/Deep360/depth_on_1', curStage='training', shuffleOrder=False, inputRGB=True, needMask=False):
    self.rootInputDir = rootInputDir
    self.rootDepthDir = rootDepthDir
    self.curStage = curStage
    self.camPairs = ['12', '13', '14', '23', '24', '34']
    self.fusionList = self.__getFileList()
    print("fusion dataset. Read saved depth maps as input")
    print("inputs Dir: {}, depth GT Dir: {}".format(self.rootInputDir, self.rootDepthDir))
    print("current stage: {}, num of files: {}".format(self.curStage, len(self.fusionList)))
    if len(self.fusionList) == 0:
      raise ValueError("Error in initializing dataset: Empty File list!")
    self.processed = preprocess.get_transform(augment=False)
    self.shuffleOrder = shuffleOrder
    self.inputRGB = inputRGB
    self.rgbImgId = [0, 5]

  def __getitem__(self, index):
    name = self.fusionList[index]
    inputDepthMaps = []
    rgbImgs = []
    r = list(range(len(self.camPairs)))
    if self.shuffleOrder:
      random.shuffle(r)
    for i in r:
      inputName = name[0][i]
      inputD = np.load(inputName).astype(np.float32)
      inputD = torch.from_numpy(inputD).unsqueeze_(0)
      inputDepthMaps.append(inputD)
    if self.inputRGB:
      for i in self.rgbImgId:
        self.processed = preprocess.get_transform(augment=False)
        leftImg = Image.open(name[2][i][0]).convert('RGB')
        rightImg = Image.open(name[2][i][1]).convert('RGB')
        rgbImgs.append(self.processed(leftImg))
        rgbImgs.append(self.processed(rightImg))
      rgbImgs = torch.cat(rgbImgs, dim=0)
    inputDepthMaps = torch.cat(inputDepthMaps, dim=0)
    depthName = name[1]
    depthMap = np.load(depthName).astype(np.float32)
    depthMap = torch.from_numpy(depthMap).unsqueeze_(0)
    data = {'inputs': inputDepthMaps, 'depthMap': depthMap, 'depthName': name[1]}
    if self.inputRGB:
      data['rgbImgs'] = rgbImgs
    return data

  def __len__(self):
    return len(self.fusionList)

  def __getFileList(self):
    fileNameList = []
    #subsets = os.listdir(self.rootDepthDir)
    for sub in scenes:
      gtDir = os.path.join(self.rootDepthDir, sub, self.curStage, 'depth')
      fileNames = sorted(os.listdir(gtDir))
      for fn in fileNames:
        fileNameList.append([os.path.join(sub, self.curStage), fn.split('_')[0]])
    fusionList = []
    for fn in fileNameList:
      dRgb = os.path.join(self.rootDepthDir, fn[0], 'rgb')
      dInput = os.path.join(self.rootInputDir, fn[0], 'pred_depth')
      dGt = os.path.join(self.rootDepthDir, fn[0], 'depth')
      depthName = os.path.join(dGt, fn[1] + '_depth.npy')
      fusion = []
      rgb = []
      for cp in self.camPairs:
        l = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[0] + '.png')
        r = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[1] + '.png')
        rgb.append([l, r])
        fusion.append(os.path.join(dInput, fn[1] + '_' + cp + '_pred_depth.npy'))
      fusionList.append([fusion.copy(), depthName, rgb.copy()])
    return fusionList


# Dataloader for soiled data
class Multi360DepthSoiledDataset(Dataset):
  def __init__(self, dataUsage='disparity', crop=False, rootDir='../../datasets/Deep360/depth_on_1', curStage='training', shape=(1024, 512), catEquiInfo=False):
    assert dataUsage in ['disparity', 'fusion']
    assert curStage in splits
    self.dataUsage = dataUsage
    self.crop = crop
    self.rootDir = rootDir
    self.curStage = curStage
    self.height, self.width = shape
    self.catEquiInfo = catEquiInfo
    self.camPairs = ['12', '13', '14', '23', '24', '34']
    self.fileNameList, self.dispList, self.fusionList = self.__getFileList()  #file name list
    #print(self.fileNameList)
    print("current stage: {}, num of frames: {}, num of disp: {}, num of fusion: {}".format(self.curStage, len(self.fileNameList), len(self.dispList), len(self.fusionList)))
    if len(self.fileNameList) == 0 or len(self.dispList) == 0 or len(self.fusionList) == 0:
      raise ValueError("Error in initializing dataset: Empty File list!")
    # element in self.fileNameList: [root, name prefix]
    # element in self.dispList: [left name, right name, disp name]
    # element in self.fusionList: [[left names of cam pairs],[right names of cam pairs],depth name]
    self.processed = preprocess.get_transform(augment=False)
    if self.catEquiInfo:
      self.equi_info = preprocess.createAngleInfo(self.height, self.width, 'X')

  def __getitem__(self, index):
    if self.dataUsage == 'disparity':
      name = self.dispList[index]
      leftName = name[0]
      rightName = name[1]
      dispName = name[2]
      leftImg = Image.open(leftName).convert('RGB')
      rightImg = Image.open(rightName).convert('RGB')
      dispMap = np.load(dispName).astype(np.float32)
      dispMap = np.ascontiguousarray(dispMap, dtype=np.float32)
      cp = name[-2:]
      if self.crop:
        w, h = leftImg.size
        th, tw = 512, 256

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
        data = {'leftImg': leftImg, 'rightImg': rightImg, 'dispMap': dispMap, 'leftName': leftName}
      else:
        leftImg = self.processed(leftImg)
        rightImg = self.processed(rightImg)
        if self.catEquiInfo:
          eq = self.equi_info
          leftImg = torch.cat([leftImg, eq], dim=0)
          rightImg = torch.cat([rightImg, eq], dim=0)
        dispMap = torch.from_numpy(dispMap).unsqueeze_(0)
        data = {'leftImg': leftImg, 'rightImg': rightImg, 'dispMap': dispMap, 'leftName': leftName}
    else:
      name = self.fusionList[index]
      imgPairs = []
      leftNames = []
      for i in range(len(self.camPairs)):
        leftName = name[0][i]
        leftNames.append(leftName)
        rightName = name[1][i]
        leftImg = Image.open(leftName).convert('RGB')
        rightImg = Image.open(rightName).convert('RGB')
        leftImg = self.processed(leftImg)
        rightImg = self.processed(rightImg)
        if self.catEquiInfo:
          eq = self.equi_info
          leftImg = torch.cat([leftImg, eq], dim=0)
          rightImg = torch.cat([rightImg, eq], dim=0)
        imgPairs.append([leftImg, rightImg])
      depthName = name[2]
      depthMap = np.load(depthName).astype(np.float32)
      depthMap = torch.from_numpy(depthMap).unsqueeze_(0)
      data = {'imgPairs': imgPairs, 'depthMap': depthMap, 'leftNames': leftNames}
    return data

  def __len__(self):
    if self.dataUsage == 'disparity':
      return len(self.dispList)
    else:
      return len(self.fusionList)

  def __getFileList(self):
    fileNameList = []
    dispList = []
    fusionList = []
    if self.curStage == 'testing':
      for sub in scenes:
        for st in soiledType:
          for sn in soiledNum:
            for spot in spotNum:
              for p in percent:
                names = os.listdir(os.path.join(self.rootDir, sub, self.curStage, 'rgb_soiled', st, sn, spot, p))
                files = set()
                for nn in names:
                  files.add(os.path.join(st, sn, spot, p, nn.split('_')[0]))
                for nn in files:
                  fileNameList.append([os.path.join(sub, self.curStage), nn])
    else:
      for sub in scenes:
        gtDir = os.path.join(self.rootDir, sub, self.curStage, 'depth')
        fileNames = sorted(os.listdir(gtDir))
        for fn in fileNames:
          fileNameList.append([os.path.join(sub, self.curStage), fn.split('_')[0]])
    for fn in fileNameList:
      nn = fn[1].split('/')[-1]
      dRgb = os.path.join(self.rootDir, fn[0], 'rgb_soiled')
      dDisp = os.path.join(self.rootDir, fn[0], 'disp')
      dDepth = os.path.join(self.rootDir, fn[0], 'depth')
      fusionL = []
      fusionR = []
      depthName = os.path.join(dDepth, nn + '_depth.npy')
      for cp in self.camPairs:
        l = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[0] + '.png')
        r = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[1] + '.png')
        disp = os.path.join(dDisp, nn + '_' + cp + '_disp.npy')
        dispList.append([l, r, disp])
        fusionL.append(l)
        fusionR.append(r)
      fusionList.append([fusionL.copy(), fusionR.copy(), depthName])
    return fileNameList, dispList, fusionList


class Multi360FusionSoiledDataset(Dataset):
  def __init__(self, rootInputDir='../../datasets/Deep360/depth_on_1', rootDepthDir='../../datasets/Deep360/depth_on_1', curStage='training', shuffleOrder=False, inputRGB=True, needMask=False):
    self.rootInputDir = rootInputDir
    self.rootDepthDir = rootDepthDir
    self.curStage = curStage
    self.camPairs = ['12', '13', '14', '23', '24', '34']
    self.processed = preprocess.get_transform(augment=False)
    self.shuffleOrder = shuffleOrder
    self.inputRGB = inputRGB
    self.needMask = needMask
    self.rgbImgId = [0, 5]
    self.fusionList = self.__getFileList()
    print("fusion dataset. Read saved depth maps as input")
    print("inputs Dir: {}, depth GT Dir: {}".format(self.rootInputDir, self.rootDepthDir))
    print("current stage: {}, num of files: {}".format(self.curStage, len(self.fusionList)))
    if len(self.fusionList) == 0:
      raise ValueError("Error in initializing dataset: Empty File list!")

  def __getitem__(self, index):
    name = self.fusionList[index]
    inputDepthMaps = []
    rgbImgs = []
    r = list(range(len(self.camPairs)))
    if self.shuffleOrder:
      random.shuffle(r)
    for i in r:
      inputName = name[0][i]
      inputD = np.load(inputName).astype(np.float32)
      inputD = torch.from_numpy(inputD).unsqueeze_(0)
      inputDepthMaps.append(inputD)
    if self.inputRGB:
      for i in self.rgbImgId:
        self.processed = preprocess.get_transform(augment=False)
        leftImg = Image.open(name[2][i][0]).convert('RGB')
        rightImg = Image.open(name[2][i][1]).convert('RGB')
        rgbImgs.append(self.processed(leftImg))
        rgbImgs.append(self.processed(rightImg))
      rgbImgs = torch.cat(rgbImgs, dim=0)
    inputDepthMaps = torch.cat(inputDepthMaps, dim=0)
    depthName = name[1]
    depthMap = np.load(depthName).astype(np.float32)
    depthMap = torch.from_numpy(depthMap).unsqueeze_(0)
    data = {'inputs': inputDepthMaps, 'depthMap': depthMap, 'depthName': name[1]}
    if self.inputRGB:
      data['rgbImgs'] = rgbImgs
    if self.curStage == 'training' and self.needMask:
      assert len(name) == 4
      maskName = name[3]
      mask = np.load(maskName).astype(np.float32)
      mask = torch.from_numpy(mask).unsqueeze_(0)
      data['soilMask'] = mask
    return data

  def __len__(self):
    return len(self.fusionList)

  def __getFileList(self):
    fileNameList = []
    fusionList = []
    if self.curStage == 'testing':
      for sub in scenes:
        for st in soiledType:
          for sn in soiledNum:
            for spot in spotNum:
              for p in percent:
                names = os.listdir(os.path.join(self.rootInputDir, sub, self.curStage, 'pred_depth_soiled', st, sn, spot, p))
                files = set()
                for nn in names:
                  files.add(os.path.join(st, sn, spot, p, nn.split('_')[0]))
                for nn in files:
                  fileNameList.append([os.path.join(sub, self.curStage), nn])

    else:
      for sub in scenes:
        gtDir = os.path.join(self.rootDepthDir, sub, self.curStage, 'depth')
        fileNames = sorted(os.listdir(gtDir))
        for fn in fileNames:
          fileNameList.append([os.path.join(sub, self.curStage), fn.split('_')[0]])
    for fn in fileNameList:
      nn = fn[1].split('/')[-1]
      dInput = os.path.join(self.rootInputDir, fn[0], 'pred_depth_soiled')
      dRgb = os.path.join(self.rootDepthDir, fn[0], 'rgb_soiled')
      dGt = os.path.join(self.rootDepthDir, fn[0], 'depth')
      depthName = os.path.join(dGt, nn + '_depth.npy')
      fusion = []
      rgb = []
      for cp in self.camPairs:
        l = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[0] + '.png')
        r = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[1] + '.png')
        rgb.append([l, r])
        fusion.append(os.path.join(dInput, fn[1] + '_' + cp + '_pred_depth_soiled.npy'))
      fusionList.append([fusion.copy(), depthName, rgb.copy()])
      if self.curStage == 'training' and self.needMask:
        maskName = os.path.join(self.rootDepthDir, fn[0], 'soil_mask', nn + '_soil_mask.npy')
        fusionList[-1].append(maskName)
    return fusionList


# Dataloader for One Frame
class Multi360FusionOneFrameDataset(Dataset):
  def __init__(self, rootInputDir='../../datasets/Deep360/depth_on_1', rootDepthDir='../../datasets/Deep360/depth_on_1', curStage='training'):
    self.rootInputDir = rootInputDir
    self.rootDepthDir = rootDepthDir
    self.curStage = curStage
    self.camPairs = ['12', '13', '14', '23', '24', '34']
    self.fusionList = self.__getFileList()
    print("fusion dataset. Read saved depth maps as input")
    print("inputs Dir: {}, depth GT Dir: {}".format(self.rootInputDir, self.rootDepthDir))
    print("current stage: {}, num of files: {}".format(self.curStage, len(self.fusionList)))
    if len(self.fusionList) == 0:
      raise ValueError("Error in initializing dataset: Empty File list!")
    self.processed = preprocess.get_transform(augment=False)

  def __getitem__(self, index):
    name = self.fusionList[index]
    inputDepthMaps = []
    inputName = name[0][0]
    inputDepthMaps = np.load(inputName).astype(np.float32)
    inputDepthMaps = torch.from_numpy(inputDepthMaps).unsqueeze_(0)
    inputDepthMaps = inputDepthMaps.repeat(6, 1, 1)
    depthName = name[1]
    depthMap = np.load(depthName).astype(np.float32)
    depthMap = torch.from_numpy(depthMap).unsqueeze_(0)
    data = {'inputs': inputDepthMaps, 'depthMap': depthMap, 'depthName': name[1]}
    return data

  def __len__(self):
    return len(self.fusionList)

  def __getFileList(self):
    fileNameList = []
    subsets = os.listdir(self.rootDepthDir)
    for sub in subsets:
      gtDir = os.path.join(self.rootDepthDir, sub, self.curStage, 'depth')
      fileNames = os.listdir(gtDir)
      for fn in fileNames:
        fileNameList.append([os.path.join(sub, self.curStage), fn.split('_')[0]])
    fusionList = []
    for fn in fileNameList:
      dInput = os.path.join(self.rootInputDir, fn[0], 'pred_depth')
      dGt = os.path.join(self.rootDepthDir, fn[0], 'depth')
      depthName = os.path.join(dGt, fn[1] + '_depth.npy')
      fusion = []
      for cp in self.camPairs:
        fusion.append(os.path.join(dInput, fn[1] + '_' + cp + '_pred_depth.npy'))
      fusionList.append([fusion.copy(), depthName])
    return fusionList


def default_loader(path):
  return Image.open(path).convert('RGB')


def disparity_loader(path):
  return np.load(path).astype(np.float32)


def depth_loader(path):

  return np.load(path).astype(np.float32)


class myImageFloder(Dataset):
  def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):

    self.left = left
    self.right = right
    self.disp_L = left_disparity
    self.loader = loader
    self.dploader = dploader
    self.training = training

  def __getitem__(self, index):
    left = self.left[index]
    right = self.right[index]
    disp_L = self.disp_L[index]

    left_img = self.loader(left)
    right_img = self.loader(right)
    disp_map = self.dploader(disp_L)
    disp_map = np.ascontiguousarray(disp_map, dtype=np.float32)

    # processed = preprocess.get_transform(augment=False)
    # left_img       = processed(left_img)
    # right_img      = processed(right_img)
    # return left_img, right_img, disp_map

    if self.training:
      w, h = left_img.size
      th, tw = 512, 256

      x1 = random.randint(0, w - tw)
      y1 = random.randint(0, h - th)

      left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
      right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

      disp_map = disp_map[y1:y1 + th, x1:x1 + tw]

      processed = preprocess.get_transform(augment=False)
      left_img = processed(left_img)
      right_img = processed(right_img)

      return left_img, right_img, disp_map
    else:
      processed = preprocess.get_transform(augment=False)
      left_img = processed(left_img)
      right_img = processed(right_img)
      return left_img, right_img, disp_map

  def __len__(self):
    return len(self.left)


class myImageFloderOnlyRGB(Dataset):
  def __init__(self, left, right, loader=default_loader, dploader=disparity_loader):

    self.left = left
    self.right = right
    self.loader = loader
    self.dploader = dploader

  def __getitem__(self, index):
    left = self.left[index]
    right = self.right[index]

    left_img = self.loader(left)
    right_img = self.loader(right)

    processed = preprocess.get_transform(augment=False)
    left_img = processed(left_img)
    right_img = processed(right_img)
    return left, left_img, right_img

  def __len__(self):
    return len(self.left)


class myImageFloderStage2(Dataset):
  def __init__(self, input12, input13, input14, input23, input24, input34, depth, training, do_disp2depth, dploader=disparity_loader, depthloader=depth_loader):

    self.input12 = input12
    self.input13 = input13
    self.input14 = input14
    self.input23 = input23
    self.input24 = input24
    self.input34 = input34
    self.depth = depth
    self.depthloader = depthloader
    self.dploader = dploader
    self.training = training
    self.do_disp2depth = do_disp2depth

  def __getitem__(self, index):
    if self.do_disp2depth:
      input12_path = self.input12[index]
      input13_path = self.input13[index]
      input14_path = self.input14[index]
      input23_path = self.input23[index]
      input24_path = self.input24[index]
      input34_path = self.input34[index]
      depth_path = self.depth[index]

      input12 = self.dploader(input12_path)
      input13 = self.dploader(input13_path)
      input14 = self.dploader(input14_path)
      input23 = self.dploader(input23_path)
      input24 = self.dploader(input24_path)
      input34 = self.dploader(input34_path)
      depth = self.depthloader(depth_path)
      depth = np.ascontiguousarray(depth, dtype=np.float32)

      depth12 = preprocess.disp2depth(input12, 0)
      depth13 = preprocess.disp2depth(input13, 1)
      depth14 = preprocess.disp2depth(input14, 2)
      depth23 = preprocess.disp2depth(input23, 3)
      depth24 = preprocess.disp2depth(input24, 4)
      depth34 = preprocess.disp2depth(input34, 5)

      np.save(input12_path[:-13].replace("disp_pred", "disp_pred2depth") + "disp_pred2depth.npy", depth12)
      np.save(input13_path[:-13].replace("disp_pred", "disp_pred2depth") + "disp_pred2depth.npy", depth13)
      np.save(input14_path[:-13].replace("disp_pred", "disp_pred2depth") + "disp_pred2depth.npy", depth14)
      np.save(input23_path[:-13].replace("disp_pred", "disp_pred2depth") + "disp_pred2depth.npy", depth23)
      np.save(input24_path[:-13].replace("disp_pred", "disp_pred2depth") + "disp_pred2depth.npy", depth24)
      np.save(input34_path[:-13].replace("disp_pred", "disp_pred2depth") + "disp_pred2depth.npy", depth34)

      return depth12, depth13, depth14, depth23, depth24, depth34, depth
    else:
      input12_path = self.input12[index]
      input13_path = self.input13[index]
      input14_path = self.input14[index]
      input23_path = self.input23[index]
      input24_path = self.input24[index]
      input34_path = self.input34[index]
      depth_path = self.depth[index]

      input12 = self.dploader(input12_path)
      input13 = self.dploader(input13_path)
      input14 = self.dploader(input14_path)
      input23 = self.dploader(input23_path)
      input24 = self.dploader(input24_path)
      input34 = self.dploader(input34_path)
      depth = self.depthloader(depth_path)
      depth = np.ascontiguousarray(depth, dtype=np.float32)

      return input12, input13, input14, input23, input24, input34, depth

    # if self.training:
    #     w, h = left_img.size
    #     th, tw = 512, 256

    #     x1 = random.randint(0, w - tw)
    #     y1 = random.randint(0, h - th)

    #     left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
    #     right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

    #     disp_map = disp_map[y1:y1 + th, x1:x1 + tw]

    #     processed = preprocess.get_transform(augment=False)
    #     left_img   = processed(left_img)
    #     right_img  = processed(right_img)

    #     return left_img, right_img, disp_map
    # else:
    #     processed = preprocess.get_transform(augment=False)
    #     left_img       = processed(left_img)
    #     right_img      = processed(right_img)
    #     return left_img, right_img, disp_map

  def __len__(self):
    return len(self.input12)


if __name__ == '__main__':
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
  da = MultiViewDeep360Dataset(rootDir='../../../datasets/Deep360/depth_on_1',
                               interDir='../outputs/depth_on_1_inter',
                               mode='intermedia_fusion',
                               curStage='training',
                               shape=(1024,
                                      512),
                               crop=False,
                               catEquiInfo=False,
                               soiled=False,
                               shuffleOrder=False,
                               inputRGB=True,
                               needMask=False)
  myDL = torch.utils.data.DataLoader(da, batch_size=1, num_workers=1, pin_memory=False, shuffle=True)
  for id, batch in enumerate(myDL):
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
    print(batch['inputs'].shape)
    print(batch['depthMap'].shape)
    print(batch['rgbImgs'].shape)
    print(batch['leftNames'])
