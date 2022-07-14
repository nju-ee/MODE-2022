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

from models import PSMstackhourglass, LCV360SD, ModesDisparity

from utils.CassiniViewTrans import *
from geometry import *

from dataloader.multi360DepthLoader import Multi360DepthDataset, Multi360DepthSoiledDataset
from dataloader.multiFisheyeLoader import OmniFisheyeDataset
from dataloader.dataset3D60Loader import Dataset3D60

parser = argparse.ArgumentParser(description='deep 360 multi view disp to depth on view 1')
parser.add_argument('--disp_model', default='Modes', help='select model')
parser.add_argument('--checkpoint_disp', default='./checkpoints/ckpt_disp_PSMNet_3d60_sphere4shg_55.tar', help='select model')
parser.add_argument("--dataset", default="deep360", choices=["deep360", "3d60", "OmniHouse", "Sunny", "Cloudy", "Sunset"], type=str, help="dataset to train on.")
parser.add_argument("--data_root_dir", default="../../datasets/3D60", type=str, help="dataset name file.")
parser.add_argument("--filelist_root", default="./dataloader", type=str, help="dataset name file.")
parser.add_argument('--widthE', default=1024, type=int, help="width of omnidirectional images in ERP domain")
parser.add_argument('--heightE', default=512, type=int, help="height of omnidirectional images in ERP domain")
parser.add_argument('--widthC', default=512, type=int, help="width of omnidirectional images in Cassini domain")
parser.add_argument('--heightC', default=1024, type=int, help="height of omnidirectional images in Cassini domain")
parser.add_argument('--batch_size', default=1, type=int, help="batch size for disp to depth trans")
# stereo
parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--max_depth', default=20, type=float, help="max valid depth")
parser.add_argument('--parallel', action='store_true', default=False, help='model parallel')
parser.add_argument('--clear_old', action='store_true', default=False, help='clear old results')
parser.add_argument('--save_root_dir', type=str, default='./outputs/output_stage1_3D60', help='model parallel')
parser.add_argument('--soiled', action='store_true', default=False, help='soiled data')

args = parser.parse_args()

scenes = ['Matterport3D', 'ep2_500frames', 'ep3_500frames', 'ep4_500frames', 'ep5_500frames', 'ep6_500frames']
splits = ['training', 'testing', 'validation']
filelists = ['3d60_train.txt', '3d60_test.txt', '3d60_val.txt']
predDirName = 'disp_pred2depth'
confDirName = 'conf_map'
depthCaDirName = 'depth'
depthErpDirName = 'depth_ERP'
rgbDirName = 'rgb'
camPairs = ['12', '21']


def checkBuildDir():
  if not os.path.exists(args.save_root_dir):
    os.makedirs(args.save_root_dir)
    for sp in splits:
      d2 = os.path.join(args.save_root_dir, sp)
      if not os.path.exists(d2):
        os.makedirs(d2)
      dp = os.path.join(d2, predDirName)
      if not os.path.exists(dp):
        os.makedirs(dp)
      dc = os.path.join(d2, confDirName)
      if not os.path.exists(dc):
        os.makedirs(dc)
      dr = os.path.join(d2, rgbDirName)
      if not os.path.exists(dr):
        os.makedirs(dr)
      dr = os.path.join(d2, depthCaDirName)
      if not os.path.exists(dr):
        os.makedirs(dr)
      dr = os.path.join(d2, depthErpDirName)
      if not os.path.exists(dr):
        os.makedirs(dr)


def clearOldFiles():
  for sc in scenes:
    d = os.path.join(args.root_dir, sc)
    for sp in splits:
      d2 = os.path.join(d, sp, predDirName)
      os.system("rm {}".format(os.path.join(d2, '*.npy')))
      #os.remove(os.path.join(d2, '*.npy'))


def disp2depth(disp, conf_map, cam_pair):
  cam_pair_dict = {'12': 0, '21': 1}

  # if args.dbname == 'Deep360':
  #   baseline = np.array([1, 1, math.sqrt(2), math.sqrt(2), 1, 1]).astype(np.float32)
  # elif args.dbname == '3D60':
  #   pass
  # else:
  #   baseline = np.array([0.6 * math.sqrt(2), 0.6 * math.sqrt(2), 1.2, 1.2, 0.6 * math.sqrt(2), 0.6 * math.sqrt(2)]).astype(np.float32)
  baseline = [0.26, 0.26]

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
  depth_l = baseline[cam_pair] * np.sin(math.pi / 2 - phi_r_map) / np.sin(phi_r_map - phi_l_map)
  depth_l = depth_l.filled(1000)
  depth_l[depth_l > 1000] = 1000
  depth_l[depth_l < 0] = 0

  if cam_pair == 0:
    return depth_l, conf_map
  elif cam_pair == 1:
    depth_2, conf_2 = depthViewTransWithConf(depth_l, conf_map, 0, 0, -0.26, math.pi, 0, 0)
    return depth_2, conf_2
  else:
    print("Error! Wrong Cam_pair!")
    return None


def batchTransAndSave(batchDisp, batchConf, cam, spid, names, max_depth=args.max_depth, cpuParallel=True):
  n, c, h, w = batchDisp.shape
  for j in range(n):
    disp_j = batchDisp[j, ::].squeeze(0).squeeze(0).cpu().numpy()
    conf_j = batchConf[j, ::].squeeze(0).squeeze(0).cpu().numpy()
    depth_j, conf_map_j = disp2depth(disp_j, conf_j, cam)
    lname = (names[j].split('/')[-1]).split('color')[0]
    #print(lname)
    savePre = lname + camPairs[cam]
    saveDepthDir = os.path.join(args.save_root_dir, splits[spid], predDirName)
    saveConfDir = os.path.join(args.save_root_dir, splits[spid], confDirName)
    np.save(os.path.join(saveDepthDir, savePre + '_' + predDirName + '.npy'), depth_j)
    cv2.imwrite(os.path.join(saveConfDir, savePre + '_' + confDirName + '.png'), conf_map_j * 255)
    #print(names[j])
    #print(os.path.join(saveDir, saveName))


def saveRGB(imgL, imgR, names, spid):
  n, c, h, w = imgL.shape
  for j in range(n):
    imgL_j = imgL[j, ::]
    imgR_j = imgR[j, ::]
    lname = (names[j].split('/')[-1]).split('color')[0]
    #print(lname)
    savePre = lname + '12_rgb'
    saveRgbDir = os.path.join(args.save_root_dir, splits[spid], rgbDirName)
    imgL_j = (imgL_j - torch.min(imgL_j)) / (torch.max(imgL_j) - torch.min(imgL_j))
    imgR_j = (imgR_j - torch.min(imgR_j)) / (torch.max(imgR_j) - torch.min(imgR_j))
    torchvision.utils.save_image(imgL_j, os.path.join(saveRgbDir, savePre + '1.png'))
    torchvision.utils.save_image(imgR_j, os.path.join(saveRgbDir, savePre + '2.png'))


def saveDepthGt(depth, depthERP, names, spid):
  n, c, h, w = depth.shape
  for j in range(n):
    depth_j = depth[j, ::].squeeze(0).squeeze(0).cpu().numpy()
    depth_erp_j = depthERP[j, ::].squeeze(0).squeeze(0).cpu().numpy()
    lname = (names[j].split('/')[-1]).split('color')[0]
    #print(lname)
    savePre = lname
    saveDepthCaDir = os.path.join(args.save_root_dir, splits[spid], depthCaDirName)
    saveDepthErpDir = os.path.join(args.save_root_dir, splits[spid], depthErpDirName)
    np.save(os.path.join(saveDepthCaDir, savePre + depthCaDirName + '.npy'), depth_j)
    np.save(os.path.join(saveDepthErpDir, savePre + depthErpDirName + '.npy'), depth_erp_j)


if __name__ == '__main__':
  checkBuildDir()
  if args.clear_old:
    clearOldFiles()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # dispModelDict = {'360SDNet': LCV360SD, 'PSMNet': PSMstackhourglass, 'Modes': ModesDisparity}
  # if args.disp_model in dispModelDict:
  #   disp_model = dispModelDict[args.disp_model](args.max_disp, conv='Sphere', in_height=1024, in_width=512, sphereType='Cassini', out_conf=True)
  # else:
  #   raise NotImplementedError('Required Model {} is Not Implemented!!!'.format(args.disp_model))
  disp_model = ModesDisparity(args.max_disp, conv='Sphere', in_height=512, in_width=256, sphereType='Cassini', out_conf=True)

  if args.parallel:
    disp_model = nn.DataParallel(disp_model)

  disp_model = disp_model.to(device)

  pretrained = torch.load(args.checkpoint_disp)
  disp_model.load_state_dict(pretrained['state_dict'])

  # if args.dataset == 'deep360':
  #   myDataset = Multi360DepthSoiledDataset if args.soiled else Multi360DepthDataset
  # elif args.dataset == "OmniHouse" or args.dataset == "Sunny" or args.dataset == "Cloudy" or args.dataset == "Sunset":
  #   myDataset = OmniFisheyeDataset
  # else:
  #   raise NotImplementedError("Dataset <{}> is not supported yet!".format(args.dataset))
  # allFileList = './dataloader/{}_all.txt'.format(args.filelist)
  myDataset = Dataset3D60
  # data
  for spid in range(0, 3):
    st = time.time()
    sp = splits[spid]
    filelist = os.path.join(args.filelist_root, filelists[spid])
    print("cur: {}, file list: {}".format(sp, filelist))
    dispData = myDataset(filenamesFile=filelist,
                         rootDir=args.data_root_dir,
                         interDir='outputs/3D60',
                         mode='fusion',
                         curStage='validation',
                         shape=(512,
                                256),
                         crop=False,
                         catEquiInfo=False,
                         soiled=False,
                         shuffleOrder=False,
                         inputRGB=False,
                         needMask=False,
                         camPairs=['12',
                                   '21'],
                         rgbIds=[],
                         copyFusion=False,
                         maxDepth=20.0,
                         saveOut=False)
    trainDispDataLoader = torch.utils.data.DataLoader(dispData, batch_size=args.batch_size, num_workers=1, pin_memory=False, shuffle=False)
    disp_model.eval()
    for batchIdx, batchData in enumerate(tqdm(trainDispDataLoader, desc='save pred trans depth maps')):
      images = batchData['imgPairs']
      leftNames = batchData['leftNames']
      dispInvalidMask = batchData['dispMask'] if myDataset == OmniFisheyeDataset else None
      depthCa = batchData['depthMap']
      depthErp = batchData['depthMapERP']
      saveDepthGt(depthCa, depthErp, leftNames[0], spid)
      n = len(leftNames[0])
      with torch.no_grad():
        for i in range(len(images)):
          leftImg, rightImg = images[i][0].to(device), images[i][1].to(device)
          disp_pred, conf_map = disp_model(leftImg, rightImg)
          if i == 0:
            saveRGB(leftImg, rightImg, leftNames[i], spid)
          if dispInvalidMask is not None:
            invMask = dispInvalidMask[i].to(device)
            disp_pred[invMask] = 0
          batchTransAndSave(disp_pred, conf_map, i, spid, leftNames[i])
    print(time.time() - st)
