import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import random
from PIL import Image
import numpy as np
import scipy.ndimage
import cv2

if __name__ == '__main__':
  import preprocess
else:
  from . import preprocess
"""
NOTE:

Multi view fisheye depth Dataset

proposed in Won et al OmniMVS: End-to-End Learning for Omnidirectional Stereo Matching, ICCV 2019
3000 scene points
each point contains 6 pairs of left-right cassini preojection images
(12,13,14,23,24,34), 6 disparity maps , 1 depth maps on center,1 ERP depth map on original view

indoor: OmniHouse
outdoor: urban (sunny, cloudy, snset)

Dataset directory structure:
OmniHouse
 - training
 - testing
 - cassini masks
 - (depth GT ori)
Sunny
Cloudy
Sunset

"""

datasetsNames = ['OmniHouse', 'Sunny', 'Cloudy', 'Sunset']
splits = ['training', 'testing']
dataType = ['rgb', 'disp', 'depth']

dataModes = ['disparity', 'fusion', 'intermedia_fusion']
inputCamPairs = ['12', '13', '14', '23', '24', '34']
inputRGBImgIds = [0, -1]


def invdepthToIndex(inv_depth, min_invdepth, sample_step_invdepth, start_index=0):
  return (inv_depth - min_invdepth) / sample_step_invdepth + start_index


def loadGTInvdepthIndex(gt_inv_depth, min_invdepth, sample_step_invdepth, remove_gt_noise=True, morph_win_size=5, invdepTh=1e-3):
  gt_idx = invdepthToIndex(gt_inv_depth, min_invdepth, sample_step_invdepth, start_index=0)
  if not remove_gt_noise:
    return gt_idx
  # make valid mask
  morph_filter = np.ones((morph_win_size, morph_win_size), dtype=np.uint8)
  finite_depth = gt_inv_depth >= invdepTh  # <= 1000 m
  closed_depth = scipy.ndimage.binary_closing(finite_depth, morph_filter)
  infinite_depth = np.logical_not(finite_depth)
  infinite_hole = np.logical_and(infinite_depth, closed_depth)
  gt_idx[infinite_hole] = -1
  gt_depth = 1.0 / gt_inv_depth
  gt_depth[~finite_depth] = 0.0
  return gt_idx, gt_depth, finite_depth


def loadTiffDepth(path):
  EPS = 3e-10
  min_depth = 0.5  # meter scale
  max_depth = 3e10
  min_invdepth = 1.0 / max_depth
  max_invdepth = 1.0 / min_depth
  num_invdepth = 192
  sample_step_invdepth = (max_invdepth - min_invdepth) / (num_invdepth - 1.0)
  multi_image = Image.open(path)
  num_read_images = multi_image.n_frames
  assert num_read_images == 2  #
  mimgs = []
  for i in range(num_read_images):
    multi_image.seek(i)
    mimgs.append(np.array(multi_image))
  if mimgs[0].dtype == np.uint8:
    invDepth = mimgs[1].squeeze()
  else:
    invDepth = mimgs[0].squeeze()
  gtIndex, gtDepth, mask = loadGTInvdepthIndex(invDepth, min_invdepth, sample_step_invdepth)
  return invDepth, gtIndex, gtDepth, mask


# TODO: test this dataset and fix bugs
# try to build one Dataset Class to implement reading data of disparity, fusion, intermedia_fusion, soiled data
class OmniFisheyeDataset(Dataset):
  def __init__(self,
               rootDir='../../datasets/Deep360/',
               interDir='../',
               depthOriDir=None,
               mode='disparity',
               curStage='training',
               datasetName='OmniHouse',
               shape=(640,
                      320),
               crop=False,
               catEquiInfo=False,
               soiled=False,
               shuffleOrder=False,
               inputRGB=True,
               needMask=False,
               camPairs=inputCamPairs,
               rgbIds=inputRGBImgIds,
               withConf=True):
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
    super(OmniFisheyeDataset, self).__init__()
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
    self.rootDir = os.path.join(rootDir, datasetName)
    self.interDir = os.path.join(interDir, 'MODE-Net_output_stage1_OmniHouse/')
    self.depthOriDir = os.path.join(self.rootDir, 'omnidepth_gt_640') if depthOriDir is None else depthOriDir
    self.curStage = curStage
    self.height, self.width = shape
    self.catEquiInfo = catEquiInfo
    self.crop = crop
    self.soiled = soiled
    self.shuffleOrder = shuffleOrder
    self.inputRGB = inputRGB
    self.needMask = needMask
    self.camPairs = camPairs
    self.withConf = withConf
    self.maskCassiniName = 'mask_cassini'
    self.processed = preprocess.get_transform(augment=False)  # transform of rgb images
    if self.inputRGB:
      self.rgbId = rgbIds  # use which pairs of RGB images
    # get file names
    self.fileNameList = self.__getFileList()
    self.dispMask = self.__getDispMask()
    print("Dataset {}: Multi-views fish eye dataset. Num of files: {}".format(datasetName, len(self.fileNameList)))
    print("root dir: {}. intermedia dir: {}. depth ori dir: {}".format(self.rootDir, self.interDir, self.depthOriDir))
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
    gtDir = os.path.join(self.rootDir, self.curStage, 'depth')
    fileNames = sorted(os.listdir(gtDir))
    for fn in fileNames:
      fileNameList.append([self.curStage, fn.split('_')[0]])
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
        dInput = os.path.join(self.interDir, fn[0], 'disp_pred2depth')
      dConf = os.path.join(self.interDir, fn[0], 'conf_map')
      dGt = os.path.join(self.rootDir, fn[0], 'depth')
      depthName = os.path.join(dGt, numsuf + '_depth.npy')
      depthOriName = os.path.join(self.depthOriDir, '0' + numsuf + '.tiff')  # tiff format original invDepth file
      fusion = []
      rgb = []
      conf_map = []
      for cp in self.camPairs:
        l = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[0] + '.png')
        r = os.path.join(dRgb, fn[1] + '_' + cp + '_rgb' + cp[1] + '.png')
        rgb.append([l, r])
        if self.soiled:
          fusion.append(os.path.join(dInput, fn[1] + '_' + cp + '_pred_depth_soiled.npy'))
        else:
          fusion.append(os.path.join(dInput, fn[1] + '_' + cp + '_disp_pred2depth.npy'))
        conf_map.append(os.path.join(dConf, fn[1] + '_' + cp + '_conf_map.png'))
      fusionList.append([fusion.copy(), depthName, rgb.copy(), depthOriName, conf_map.copy()])
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
      lmask = np.expand_dims(lmask, 0)
      rmask = np.expand_dims(rmask, 0)
      lmask = lmask / 255.0
      rmask = rmask / 255.0
      overLap = lmask + rmask
      invalid = (overLap > 0)
      m = [lmask, rmask, torch.from_numpy(invalid)]
      # invalidSave = invalid * 255
      # invalidSave = invalidSave.astype(np.uint8)
      # cv2.imwrite('invalidMask_' + cp + '.png', invalidSave)
      masks.append(torch.from_numpy(invalid))
    return masks

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
    cpid = self.camPairs.index(cp)
    invalidMask = self.dispMask[cpid]
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
      data = {'leftImg': leftImg, 'rightImg': rightImg, 'dispMap': dispMap, 'leftNames': leftName, 'invalidMask': invalidMask}
    return data

  def __getFusionData(self, index):  # return data in depth fusion (full stage) task
    name = self.fusionList[index]
    imgPairs = []
    leftNames = []
    dMask = self.dispMask.copy()
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
    data = {'imgPairs': imgPairs, 'depthMap': depthMap, 'leftNames': leftNames, 'dispMask': dMask}
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
      if self.withConf:
        inputC = (cv2.imread(name[4][i], cv2.IMREAD_GRAYSCALE).astype(np.float32)) / 255.0
        inputC = torch.from_numpy(inputC).unsqueeze_(0)
        inputDepthMaps.append(inputC)
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
    invDepth, gtIndexERP, gtDepthERP, finiteMask = loadTiffDepth(name[3])
    data = {'inputs': inputDepthMaps, 'depthMap': depthMap, 'leftNames': name[0], 'depthName': name[1], 'gtIndexERP': gtIndexERP}  # return index as well
    if self.inputRGB:
      data['rgbImgs'] = rgbImgs
    if self.curStage == 'training' and self.needMask:
      assert len(name) == 5
      maskName = name[4]
      mask = np.load(maskName).astype(np.float32)
      mask = torch.from_numpy(mask).unsqueeze_(0)
      data['soilMask'] = mask
    return data


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
  da = OmniFisheyeDataset(rootDir='../../../datasets/Deep360/',
                          interDir='../outputs/OmniHouse_inter',
                          mode='disparity',
                          curStage='training',
                          datasetName='OmniHouse',
                          shape=(640,
                                 320),
                          crop=False,
                          catEquiInfo=False,
                          soiled=False,
                          shuffleOrder=False,
                          inputRGB=True,
                          needMask=False)
  myDL = torch.utils.data.DataLoader(da, batch_size=1, num_workers=1, pin_memory=False, shuffle=True, drop_last=False)
  maxDisp = 0
  for id, batch in enumerate(tqdm(myDL, desc='Train iter')):
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
    # print(batch['dispMap'].shape)
    # print(batch['invalidMask'].shape)
    # print(batch['leftNames'])
    disp = batch['dispMap']
    disp[torch.isnan(disp)] = 0
    #print(torch.max(disp), torch.min(disp))
    if (torch.max(disp) > 192):
      print(batch['leftNames'])
    maxDisp = max(maxDisp, torch.max(disp))
  print(maxDisp)
