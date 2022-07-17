from __future__ import print_function
import os

import argparse
from sqlalchemy import true
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

from models import ModeDisparity

import dataloader.preprocess as preprocess

parser = argparse.ArgumentParser(description='Multi View Omnidirectional Depth Estimation')

# model
parser.add_argument('--disp_model', default='MODE', help='select model')
parser.add_argument('--fusion_model', default='withRGB', help='select fusion model')
# data
parser.add_argument("--stage", default="disp", choices=["disp", "all"], type=str, help="stage")
parser.add_argument("--data_root", default="../tmp", type=str, help="dataset root directory.")
parser.add_argument("--save_root", default="../tmp", type=str, help="dataset root directory.")

parser.add_argument('--widthE', default=1024, type=int, help="width of omnidirectional images in ERP domain")
parser.add_argument('--heightE', default=512, type=int, help="height of omnidirectional images in ERP domain")
parser.add_argument('--widthC', default=512, type=int, help="width of omnidirectional images in Cassini domain")
parser.add_argument('--heightC', default=1024, type=int, help="height of omnidirectional images in Cassini domain")

# for multi-view
parser.add_argument('--img_name_prefix', default='ca', type=str, help="prefix of multi-view inputs")
parser.add_argument('--img_type', default='.png', type=str, help="suffix (extension name) of inputs")
# for left-right stereo
parser.add_argument('--left_name', default='', type=str, help="left name")
parser.add_argument('--right_name', default='', type=str, help="right name")

parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--max_depth', default=1000, type=float, help="max valid depth")

# model
parser.add_argument('--checkpoint_disp', default='./checkpoints/ckpt_disp_PSMNet_30.tar', help='load checkpoint of disparity estimation path')
parser.add_argument('--checkpoint_fusion', default='./checkpoints/saved/ckpt_fusion_Unet_30-sie.tar', help='load checkpoint of fusion module path')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--parallel', action='store_true', default=False, help='train model parallel')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
# predict for real scene

model_disp = ModeDisparity(maxdisp=args.max_disp, conv='Sphere', in_height=args.height, in_width=args.width, sphereType='Cassini', out_conf=True)
model_disp.eval()
if (args.parallel):
  model_disp = nn.DataParallel(model_disp)
if args.cuda:
  model_disp.cuda()
if (args.checkpoint_disp is not None):
  state_dict = torch.load(args.checkpoint_disp)
  model_disp.load_state_dict(state_dict['state_dict'])
else:
  raise ValueError("disp model checkpoint is not defined")

processed = preprocess.get_transform(augment=False)

if args.stage == 'disp':
  saveLogColor = True
  with torch.no_grad():
    left = Image.open(args.left_name).convert('RGB')
    right = Image.open(args.right_name).convert('RGB')
    left = processed(left).unsqueeze(0)
    right = processed(right).unsqueeze(0)
    pred_disp, pred_conf = model_disp(left, right)
    pred_disp[pred_disp < 0] = 0
    pred_disp_save = pred_disp.squeeze().cpu().numpy()
    if saveLogColor:
      pred_disp_save = np.log(pred_disp_save + 1.0)
    pred_disp_save = ((pred_disp_save - np.min(pred_disp_save)) / (np.max(pred_disp_save) - np.min(pred_disp_save)) * 255).astype(np.uint8)
    pred_disp_save = cv2.applyColorMap(pred_disp_save, cv2.COLORMAP_JET)
    cv2.imwrite("output_disp.png", pred_disp_save)
    pred_conf_save = pred_conf.squeeze().cpu().numpy() * 255
    cv2.imwrite("output_conf.png", pred_conf_save)
