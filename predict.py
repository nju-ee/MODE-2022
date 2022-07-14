from __future__ import print_function
import os

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
from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models import PSMstackhourglass, LCV360SD, AANet, ModesDisparity
from models import Fusion0, FusionUnet, FusionDilate, FusionWithRGB
from utils.dispAndDepth import DispDepthTransformerCassini as DDTC
from utils.ERPandCassini import CA2ERP
import utils.projection as projection
from utils.CassiniViewTrans import *
from utils.loss import scaleInvariantLoss, psmnetSHGLoss, aanetMultiScaleLoss

from dataloader.dataset3D60Loader import Dataset3D60
from dataloader.multi360DepthLoader import Multi360DepthDataset, Multi360FusionDataset

import dataloader.preprocess as preprocess

parser = argparse.ArgumentParser(description='Multi View Omnidirectional Depth Estimation')

# model
parser.add_argument('--disp_model', default='Modes', help='select model')
parser.add_argument('--fusion_model', default='withRGB', help='select fusion model')
# data
parser.add_argument("--dataset", default="carla", choices=["carla", "3d60", "panosuncg", "stanford2d3d", "matterport3d"], type=str, help="dataset to train on.")
parser.add_argument("--data_root", default="../tmp", type=str, help="dataset root directory.")
parser.add_argument("--save_root", default="../tmp", type=str, help="dataset root directory.")
parser.add_argument("--filelist", default="deep360", choices=["deep360", "suncg", "mat3d", "3d60", "panosuncg", "stanford2d3d", "matterport3d"], type=str, help="dataset name file.")
parser.add_argument('--intermedia_path',
                    default='./outputs/depth_on_1_inter',
                    help='intermedia results saving path. directory to save predict depth maps transformed form disparity maps using for fusion')
parser.add_argument('--widthE', default=1024, type=int, help="width of omnidirectional images in ERP domain")
parser.add_argument('--heightE', default=512, type=int, help="height of omnidirectional images in ERP domain")
parser.add_argument('--widthC', default=512, type=int, help="width of omnidirectional images in Cassini domain")
parser.add_argument('--heightC', default=1024, type=int, help="height of omnidirectional images in Cassini domain")
parser.add_argument('--data_struct', default='suffix', type=str, choices=['suffix', 'subdir'], help="height of omnidirectional images in Cassini domain")
parser.add_argument('--img_name', default='ca', type=str, help="height of omnidirectional images in Cassini domain")
parser.add_argument('--img_type', default='.png', type=str, help="height of omnidirectional images in Cassini domain")
# stereo
parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--max_depth', default=1000, type=float, help="max valid depth")
parser.add_argument('--baseline', default=1, type=float, help="baseline of binocular spherical system")

# model
parser.add_argument('--checkpoint_disp', default='./checkpoints/ckpt_disp_PSMNet_30.tar', help='load checkpoint of disparity estimation path')
parser.add_argument('--checkpoint_conf', default=None, help='load checkpoint of confidence estimation path')
parser.add_argument('--checkpoint_fusion', default='./checkpoints/saved/ckpt_fusion_Unet_30-sie.tar', help='load checkpoint of fusion module path')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--printNet', action='store_true', default=False, help='print network details')

parser.add_argument('--parallel_disp', action='store_true', default=False, help='train model parallel')
parser.add_argument('--parallel_fusion', action='store_true', default=False, help='train model parallel')
parser.add_argument('--cudnn_deter', action='store_true', default=False, help='if True, set cudnn deterministic as True and benchmark as False. Otherwise the opposite')
parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 123)')
parser.add_argument("--view_trans", default="point_gather", choices=["grid_sample", "point_gather"], type=str, help="method for view transformation")

# saving
parser.add_argument('--save_suffix_disp', type=str, default=None, help='save checkpoint name')
parser.add_argument('--save_suffix_fusion', type=str, default=None, help='save checkpoint name')
parser.add_argument('--save_checkpoint_path', default='./checkpoints', help='save checkpoint path')
parser.add_argument('--save_image_path', type=str, default='./outputs', help='save images path')

args = parser.parse_args()
realCamConfigs = {
    'height':
    1024,
    'width':
    512,
    'cam_pair_num':
    6,
    'baselines': [0.3,
                  0.3,
                  0.3 * math.sqrt(2),
                  0.3 * math.sqrt(2),
                  0.3,
                  0.3],
    'trans_rotate': [
        #y,z,x,pitch,yaw,roll
        #12
        [0,
         0,
         0,
         0,
         0,
         0],
        #13
        [0,
         0,
         0,
         0.5 * math.pi,
         0,
         0],
        #14
        [0,
         0,
         0,
         0.25 * math.pi,
         0,
         0],
        #23
        [0,
         -math.sqrt(2) / 2,
         -math.sqrt(2) / 2,
         0.75 * math.pi,
         0,
         0],
        #24
        [0,
         -1,
         0,
         0.5 * math.pi,
         0,
         0],
        #34
        [0,
         1,
         0,
         0,
         0,
         0]
    ]
}
subdirs = ['NEK4Q8', 'N4FFDR', 'N4KKVT', 'NF38XS']
camPairs = ['12', '13', '14', '23', '24', '34']
dispModelDict = {'360SDNet': LCV360SD, 'PSMNet': PSMstackhourglass, 'AANet': AANet, 'Modes': ModesDisparity}
fusionModelDict = {'Fusion0': Fusion0, 'Unet': FusionUnet, 'Dilate': FusionDilate, 'withRGB': FusionWithRGB}
if args.disp_model in dispModelDict:
  if args.disp_model == 'Modes':
    disp_model = dispModelDict[args.disp_model](args.max_disp, conv='Sphere', in_height=1024, in_width=512, sphereType='Cassini', out_conf=True)
  else:
    disp_model = dispModelDict[args.disp_model](args.max_disp)
else:
  raise NotImplementedError('Required Model {} is Not Implemented!!!'.format(args.disp_model))

if args.fusion_model in fusionModelDict:
  fusion_model = fusionModelDict[args.fusion_model](max_depth=args.max_depth)
else:
  raise NotImplementedError('Required Model {} is Not Implemented!!!'.format(args.fusion_model))

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
  disp_model = disp_model.cuda()
  fusion_model = fusion_model.cuda()
if args.parallel_disp:
  disp_model = nn.DataParallel(disp_model)
if args.parallel_fusion:
  fusion_model = nn.DataParallel(fusion_model)

if (args.checkpoint_disp is not None) and (args.checkpoint_disp != 'None'):
  state_dict = torch.load(args.checkpoint_disp)
  disp_model.load_state_dict(state_dict['state_dict'])
else:
  raise ValueError("disp model checkpoint is not defined")

if (args.checkpoint_fusion is not None) and (args.checkpoint_fusion != 'None'):
  state_dict = torch.load(args.checkpoint_fusion)
  fusion_model.load_state_dict(state_dict['state_dict'])
else:
  raise ValueError("fusion model checkpoint is not defined")
ca2e = CA2ERP(args.heightE, args.widthE, args.heightC, args.widthC)
imgNames = []
if args.data_struct == 'suffix':
  for cp in camPairs:
    imgNames.append([args.img_name + '_' + cp + '_rgb' + cp[0] + args.img_type, args.img_name + '_' + cp + '_rgb' + cp[1] + args.img_type])
processed = preprocess.get_transform(augment=False)
left = []
right = []
for i in range(6):
  l = Image.open(os.path.join(args.data_root, imgNames[i][0])).convert('RGB')
  l = l.resize((args.widthC, args.heightC), Image.ANTIALIAS)
  left.append(processed(l).unsqueeze_(0))
  r = Image.open(os.path.join(args.data_root, imgNames[i][1])).convert('RGB')
  r = r.resize((args.widthC, args.heightC), Image.ANTIALIAS)
  right.append(processed(r).unsqueeze_(0))
left = torch.cat(left, dim=0)
right = torch.cat(right, dim=0)
print(left.shape)
if args.cuda:
  left = left.cuda()
  right = right.cuda()
disp_model.eval()
fusion_model.eval()
with torch.no_grad():
  #dispPred = disp_model(left, right)
  depthMaps = []
  for i in range(6):
    leftImg = left[i:i + 1, ::]
    rightImg = right[i:i + 1, ::]
    if i == 0:
      leftErp = ca2e.trans(leftImg, '0')
    dispi, probi = disp_model(leftImg, rightImg)
    dispi[dispi > args.max_disp] = 0
    #dispi = dispPred[i:i + 1, ::]
    dispisave = torch.log10(dispi + 1.0)
    dispisave = dispisave.squeeze_(0).squeeze_(0).cpu().numpy()
    dispisave = (dispisave - np.min(dispisave)) / (np.max(dispisave) - np.min(dispisave)) * 255
    dispisave = dispisave.astype(np.uint8)
    dispisave = cv2.applyColorMap(dispisave, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.save_root, args.img_name + '_disp_' + str(i) + '.png'), dispisave)

    probisave = probi.squeeze_(0).squeeze_(0).cpu().numpy() * 255
    probisave = probisave.astype(np.uint8)
    cv2.imwrite(os.path.join(args.save_root, args.img_name + '_conf_' + str(i) + '.png'), probisave)
    depthMaps.append(batchViewTransOri(dispi, i, configs=realCamConfigs))

  # depthMaps = torch.cat(depthMaps, dim=1)

  # pred = fusion_model(depthMaps)
  # predErp = ca2e.trans(pred, '0')
  # predErpSave = torch.log10(predErp + 1.0)
  # predErp = predErp.squeeze_(0).squeeze_(0).cpu().numpy()
  # cv2.imwrite(os.path.join(args.data_root, args.img_name + '_predERP.exr'), predErp)
  # predErpSave = predErpSave.squeeze_(0).squeeze_(0).cpu().numpy()
  # predErpSave = (predErpSave - np.min(predErpSave)) / (np.max(predErpSave) - np.min(predErpSave)) * 255
  # predErpSave = predErpSave.astype(np.uint8)
  # cv2.imwrite(os.path.join(args.data_root, args.img_name + '_pred_depth_erp.png'), cv2.applyColorMap(predErpSave, cv2.COLORMAP_JET))
  # leftErp = (leftErp - torch.min(leftErp)) / (torch.max(leftErp) - torch.min(leftErp))
  # leftErp = leftErp.squeeze_(0).cpu().numpy() * 255
  # leftErp = leftErp.astype(np.uint8)
  # print(np.max(leftErp), np.min(leftErp))
  # print(leftErp.shape)
  # cv2.imwrite(os.path.join(args.data_root, args.img_name + '_erp1.jpg'), leftErp)
  # predori = pred.squeeze_(0).squeeze_(0).cpu().numpy()
  # cv2.imwrite(os.path.join(args.data_root, args.img_name + '_pred.exr'), predori)
  # predsave = torch.log10(pred + 1.0)
  # predsave = predsave.squeeze_(0).cpu().numpy()
  # predsave = (predsave - np.min(predsave)) / (np.max(predsave) - np.min(predsave)) * 255
  # predsave = predsave.astype(np.uint8)
  # cv2.imwrite(os.path.join(args.data_root, args.img_name + '_pred_depth.png'), cv2.applyColorMap(predsave, cv2.COLORMAP_JET))
  # torchvision.utils.save_image(dispisave, os.path.join(args.data_root, args.img_name + '_pred_depth.png'))
