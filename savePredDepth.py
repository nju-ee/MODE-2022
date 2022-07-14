from __future__ import print_function
import os
import sys

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import time
from datetime import datetime
import math
import random
import cv2
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models import PSMstackhourglass, LCV360SD

from utils.CassiniViewTrans import *

from dataloader.multi360DepthLoader import Multi360DepthDataset, Multi360DepthSoiledDataset
from dataloader.multiFisheyeLoader import OmniFisheyeDataset

parser = argparse.ArgumentParser(description='deep 360 multi view disp to depth on view 1')
parser.add_argument('--disp_model', default='PSMNet', help='select model')
parser.add_argument('--checkpoint_disp', default='./checkpoints/ckpt_disp_PSMNet_30.tar', help='select model')
parser.add_argument("--dataset", default="deep360", choices=["deep360", "3d60", "OmniHouse", "Sunny", "Cloudy", "Sunset"], type=str, help="dataset to train on.")
parser.add_argument("--filelist", default="deep360", choices=["deep360", "suncg", "mat3d", "3d60", "panosuncg", "stanford2d3d", "matterport3d"], type=str, help="dataset name file.")
parser.add_argument('--widthE', default=1024, type=int, help="width of omnidirectional images in ERP domain")
parser.add_argument('--heightE', default=512, type=int, help="height of omnidirectional images in ERP domain")
parser.add_argument('--widthC', default=512, type=int, help="width of omnidirectional images in Cassini domain")
parser.add_argument('--heightC', default=1024, type=int, help="height of omnidirectional images in Cassini domain")
parser.add_argument('--batch_size', default=1, type=int, help="batch size for disp to depth trans")
# stereo
parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--max_depth', default=1000, type=float, help="max valid depth")
parser.add_argument('--parallel', action='store_true', default=False, help='model parallel')
parser.add_argument('--clear_old', action='store_true', default=False, help='clear old results')
parser.add_argument('--root_dir', type=str, default='./outputs/depth_on_1_inter', help='model parallel')
parser.add_argument('--soiled', action='store_true', default=False, help='soiled data')

args = parser.parse_args()

scenes = ['ep1_500frames', 'ep2_500frames', 'ep3_500frames', 'ep4_500frames', 'ep5_500frames', 'ep6_500frames']
splits = ['training', 'testing', 'validation']
soiledType = ['glare', 'mud', 'water']
soiledNum = ['1_soiled_cam', '2_soiled_cam']
spotNum = ['2_spot', '3_spot', '4_spot', '5_spot', '6_spot']
percent = ['05percent', '10percent', '15percent', '20percent']
predDirName = 'pred_depth'
predDirSoiledName = 'pred_depth_soiled'


def checkBuildDir(soiled=False):
  predDir = predDirSoiledName if soiled else predDirName
  if not os.path.exists(args.root_dir):
    os.makedirs(args.root_dir)
  for sc in scenes:
    d = os.path.join(args.root_dir, sc)
    if not os.path.exists(d):
      os.makedirs(d)
    for sp in splits:
      d2 = os.path.join(d, sp)
      if not os.path.exists(d2):
        os.makedirs(d2)
      d3 = os.path.join(d2, predDir)
      if not os.path.exists(d3):
        os.makedirs(d3)
      if soiled and sp == 'testing':
        for st in soiledType:
          dst = os.path.join(d3, st)
          if not os.path.exists(dst):
            os.makedirs(dst)
          for sn in soiledNum:
            dsn = os.path.join(dst, sn)
            if not os.path.exists(dst):
              os.makedirs(dst)
            for spot in spotNum:
              dspot = os.path.join(dsn, spot)
              if not os.path.exists(dspot):
                os.makedirs(dspot)
              for p in percent:
                dp = os.path.join(dspot, p)
                if not os.path.exists(dp):
                  os.makedirs(dp)


def clearOldFiles():
  for sc in scenes:
    d = os.path.join(args.root_dir, sc)
    for sp in splits:
      d2 = os.path.join(d, sp, predDirName)
      os.system("rm {}".format(os.path.join(d2, '*.npy')))
      #os.remove(os.path.join(d2, '*.npy'))


def batchTransAndSave(batchDisp, camPair, names, curStage, soiled, max_depth=1000, cpuParallel=True):
  n, c, h, w = batchDisp.shape
  tmp = []
  predDir = predDirSoiledName if args.soiled else predDirName
  for j in range(n):
    disp_j = batchDisp[j, ::].squeeze(0).squeeze(0).cpu().numpy()
    depth_j = disp2depth(disp_j, camPair, max_depth, cpuParallel)

    if args.soiled and curStage == 'testing':
      lname = names[j].split('/')
      saveDir = os.path.join(
          args.root_dir,
          lname[-8],
          lname[-7],
          predDir,
          lname[-5],
          lname[-4],
          lname[-3],
          lname[-2],
      )
    else:
      lname = names[j].split('/')
      saveDir = os.path.join(args.root_dir, lname[-4], lname[-3], predDir)
    saveName = lname[-1][:-8] + predDir + '.npy'
    np.save(os.path.join(saveDir, saveName), depth_j)
    #print(names[j])
    #print(os.path.join(saveDir, saveName))


def checkBuildDirFisheye():
  predDir = predDirName
  if not os.path.exists(args.root_dir):
    os.makedirs(args.root_dir)
  d1 = os.path.join(args.root_dir, args.dataset)
  if not os.path.exists(d1):
    os.makedirs(d1)
  d2 = os.path.join(args.root_dir, args.dataset)
  if not os.path.exists(d2):
    os.makedirs(d2)
  for sp in ['training', 'testing']:
    d3 = os.path.join(d2, sp)
    if not os.path.exists(d3):
      os.makedirs(d3)
    d4 = os.path.join(d3, predDir)
    if not os.path.exists(d4):
      os.makedirs(d4)


def batchTransAndSaveFisheye(batchDisp, camPair, names, curStage, soiled, max_depth=1000, cpuParallel=True):
  n, c, h, w = batchDisp.shape
  tmp = []
  predDir = predDirSoiledName if args.soiled else predDirName
  for j in range(n):
    disp_j = batchDisp[j, ::].squeeze(0).squeeze(0).cpu().numpy()
    depth_j = disp2depth(disp_j, camPair, max_depth, cpuParallel, configs=omniFisheyeConfigs)
    lname = names[j].split('/')
    saveDir = os.path.join(args.root_dir, lname[-4], lname[-3], predDir)
    saveName = lname[-1][:-8] + predDir + '.npy'
    np.save(os.path.join(saveDir, saveName), depth_j)
    #print(names[j])
    #print(os.path.join(saveDir, saveName))


def saveData(predDepth, names):
  b, c, h, w = predDepth.shape
  predDir = predDirSoiledName if args.soiled else predDirName
  for i in range(b):
    dep = predDepth[i, ::].squeeze(0).squeeze(0).cpu().numpy()
    lname = names[i].split('/')
    saveDir = os.path.join(args.root_dir, lname[-4], lname[-3], predDir)
    saveName = lname[-1][:-8] + predDir + '.npy'
    np.save(os.path.join(saveDir, saveName), dep)


if __name__ == '__main__':
  if args.dataset == 'deep360':
    checkBuildDir(soiled=args.soiled)
    if args.clear_old:
      clearOldFiles()
  elif args.dataset == "OmniHouse" or args.dataset == "Sunny" or args.dataset == "Cloudy" or args.dataset == "Sunset":
    checkBuildDirFisheye()
  else:
    raise NotImplementedError("Dataset <{}> is not supported yet!".format(args.dataset))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  dispModelDict = {'360SDNet': LCV360SD, 'PSMNet': PSMstackhourglass}
  if args.disp_model in dispModelDict:
    disp_model = dispModelDict[args.disp_model](args.max_disp)
  else:
    raise NotImplementedError('Required Model {} is Not Implemented!!!'.format(args.disp_model))

  if args.parallel:
    disp_model = nn.DataParallel(disp_model)

  disp_model = disp_model.to(device)

  pretrained = torch.load(args.checkpoint_disp)
  disp_model.load_state_dict(pretrained['state_dict'])

  if args.dataset == 'deep360':
    myDataset = Multi360DepthSoiledDataset if args.soiled else Multi360DepthDataset
  elif args.dataset == "OmniHouse" or args.dataset == "Sunny" or args.dataset == "Cloudy" or args.dataset == "Sunset":
    myDataset = OmniFisheyeDataset
  else:
    raise NotImplementedError("Dataset <{}> is not supported yet!".format(args.dataset))
  allFileList = './dataloader/{}_all.txt'.format(args.filelist)
  # data
  for sp in splits:
    st = time.time()
    print("cur: {}".format(sp))
    if args.dataset == 'deep360':
      dispData = myDataset(dataUsage='fusion', crop=False, curStage=sp)
    elif args.dataset == "OmniHouse" or args.dataset == "Sunny" or args.dataset == "Cloudy" or args.dataset == "Sunset":
      dispData = myDataset(mode='fusion', curStage=sp, datasetName=args.dataset, shape=(640, 320), crop=False, catEquiInfo=False, soiled=False, shuffleOrder=False, inputRGB=True, needMask=False)
    trainDispDataLoader = torch.utils.data.DataLoader(dispData, batch_size=args.batch_size, num_workers=1, pin_memory=False, shuffle=False)
    disp_model.eval()
    for batchIdx, batchData in enumerate(tqdm(trainDispDataLoader, desc='save pred trans depth maps')):
      images = batchData['imgPairs']
      leftNames = batchData['leftNames']
      dispInvalidMask = batchData['dispMask'] if myDataset == OmniFisheyeDataset else None
      n = len(leftNames[0])
      with torch.no_grad():
        for i in range(len(images)):
          leftImg, rightImg = images[i][0].to(device), images[i][1].to(device)
          disp_pred = disp_model(leftImg, rightImg).detach()
          if dispInvalidMask is not None:
            invMask = dispInvalidMask[i].to(device)
            disp_pred[invMask] = 0
          if myDataset == OmniFisheyeDataset:
            batchTransAndSaveFisheye(disp_pred, i, leftNames[i], curStage=sp)
          else:
            batchTransAndSave(disp_pred, i, leftNames[i], curStage=sp, soiled=args.soiled)
    print(time.time() - st)
