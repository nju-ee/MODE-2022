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
from tensorboardX import SummaryWriter
from tqdm import tqdm
import re
import prettytable as pt

from models import ModeDisparity
from utils import evaluation, geometry
from dataloader import list_deep360_disparity_test, Deep360DatasetDisparity

parser = argparse.ArgumentParser(description='MODE Disparity estimation testing')

parser.add_argument('--model_disp', default='ModeDisparity', help='select model')
parser.add_argument("--dataset", default="Deep360", type=str, help="dataset name")

parser.add_argument("--dataset_root", default="../../datasets/Deep360/", type=str, help="dataset root directory.")
parser.add_argument('--width', default=512, type=int, help="width of omnidirectional images in Cassini domain")
parser.add_argument('--height', default=1024, type=int, help="height of omnidirectional images in Cassini domain")
# stereo
parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--max_depth', default=1000, type=float, help="max valid depth")
# hyper parameters

parser.add_argument('--batch_size', type=int, default=1, help='number of batch to train')

parser.add_argument('--checkpoint_disp', default=None, help='load checkpoint of disparity estimation path')

parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--parallel', action='store_true', default=False, help='train model parallel')

parser.add_argument('--soiled', action='store_true', default=False, help='test soiled image')
parser.add_argument('--save_output_path', type=str, default=None, help='path to save output files. if set to None, will not save')
parser.add_argument('--save_ori', action='store_true', default=False, help='save original disparity or depth value map')

args = parser.parse_args()

heightC, widthC = args.height, args.width  #Cassini shape
heightE, widthE = args.width, args.height  #ERP shape

save_out = args.save_output_path is not None
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


def saveOutputOriValue(pred, gt, mask, rootDir, id, names=None):
  b, c, h, w = pred.shape
  pred[~mask] = 0
  gt[~mask] = 0
  for i in range(b):
    predSave = pred[i, ::].cpu()
    saveimg = predSave.squeeze_(0).numpy()
    if names is None:
      name = "{:0>4}_test".format(id + i)
    else:
      oriName = names[i]
      if isinstance(oriName, list) or isinstance(oriName, tuple):
        oriName = oriName[0]
      name = os.path.splitext(os.path.basename(oriName))[0]
      if args.dataset == 'Deep360':
        ep_name = re.findall(r'ep[0-9]_', oriName)[0]
        name = ep_name + name
    np.savez(os.path.join(rootDir, name + '_pred.npz'), saveimg)


def saveOutput(pred, gt, mask, rootDir, id, names=None, log=True, savewithGt=True):
  b, c, h, w = pred.shape
  div = torch.ones([c, h, 10])
  if log:
    div = torch.log10(div * 1000 + 1.0)
    pred[mask] = torch.log10(pred[mask] + 1.0)
    gt[mask] = torch.log10(gt[mask] + 1.0)
  pred[~mask] = 0
  gt[~mask] = 0
  for i in range(b):
    predSave = pred[i, ::].cpu()
    gtSave = gt[i, ::].cpu()
    maskSave = mask[i, ::].cpu()
    if savewithGt:
      saveimg = torch.cat([gtSave, div, predSave], dim=2).squeeze_(0).numpy()
      maskSave = torch.cat([maskSave, div > 0, maskSave], dim=2).squeeze_(0).numpy()
    else:
      saveimg = predSave.squeeze_(0).numpy()
      maskSave = maskSave.squeeze_(0).numpy()
    saveimg = (saveimg - np.min(saveimg)) / (np.max(saveimg) - np.min(saveimg)) * 255

    saveimg = saveimg.astype(np.uint8)
    if names is None:
      name = "{:0>4}_test".format(id + i)
    else:
      oriName = names[i]
      if isinstance(oriName, list) or isinstance(oriName, tuple):
        oriName = oriName[0]
      name = os.path.splitext(os.path.basename(oriName))[0]
      if args.dataset == 'Deep360':
        ep_name = re.findall(r'ep[0-9]_', oriName)[0]
        name = ep_name + name
    saveimg = cv2.applyColorMap(saveimg, cv2.COLORMAP_JET)
    saveimg[:, w:w + 10, :] = 255
    saveimg[~maskSave, :] = 0
    cv2.imwrite(os.path.join(rootDir, name + '_pred.png'), saveimg)


def testDisp(modelDisp, testDispDataLoader, modelNameDisp, numTestData):
  test_metrics = ['MAE', 'RMSE', 'Px1 (%)', 'Px3 (%)', 'Px5 (%)', 'D1 (%)']
  total_eval_metrics = np.zeros(len(test_metrics))  # mae,rmse,px1,px3,px5,d1
  if save_out:
    os.makedirs(args.save_output_path, exist_ok=True)
  modelDisp.eval()
  print("Testing of Disparity. Model: {}".format(modelNameDisp))
  print("num of test files: {}".format(numTestData))
  counter = 0
  with torch.no_grad():
    for batch_idx, batchData in enumerate(tqdm(testDispDataLoader, desc='Test iter')):
      leftImg = batchData['leftImg'].to(device)
      rightImg = batchData['rightImg'].to(device)
      dispMap = batchData['dispMap'].to(device)
      b, c, h, w = leftImg.shape
      mask = (dispMap > 0) & (~torch.isnan(dispMap)) & (~torch.isinf(dispMap)) & (dispMap <= args.max_disp)
      output = modelDisp(leftImg, rightImg)
      eval_metrics = []
      eval_metrics.append(evaluation.mae(output[mask], dispMap[mask]))
      eval_metrics.append(evaluation.rmse(output[mask], dispMap[mask]))
      eval_metrics.append(evaluation.pixel_error_pct(1, output[mask], dispMap[mask]))
      eval_metrics.append(evaluation.pixel_error_pct(3, output[mask], dispMap[mask]))
      eval_metrics.append(evaluation.pixel_error_pct(5, output[mask], dispMap[mask]))
      eval_metrics.append(evaluation.D1(3, 0.05, output[mask], dispMap[mask]))
      if save_out:
        if args.save_ori: saveOutputOriValue(output.clone(), dispMap.clone(), mask, args.save_output_path, counter, names=batchData['dispNames'])  # save npz
        saveOutput(output.clone(), dispMap.clone(), mask, args.save_output_path, counter, names=batchData['dispNames'], log=True)
      total_eval_metrics += eval_metrics
    mean_errors = total_eval_metrics / len(testDispDataLoader)
    mean_errors = ['{:^.4f}'.format(x) for x in mean_errors]
  tb = pt.PrettyTable()
  tb.field_names = test_metrics
  tb.add_row(list(mean_errors))
  print('\nTest Results on Disparity using model {}:\n'.format(args.checkpoint_disp))
  print(tb)


def main():
  # model
  model_disp = ModeDisparity(maxdisp=args.max_disp, conv='Sphere', in_height=args.height, in_width=args.width, sphereType='Cassini', out_conf=False)
  if (args.parallel):
    model_disp = nn.DataParallel(model_disp)
  if args.cuda:
    model_disp.cuda()
  if (args.checkpoint_disp is not None):
    state_dict = torch.load(args.checkpoint_disp)
    model_disp.load_state_dict(state_dict['state_dict'])
  else:
    raise ValueError("disp model checkpoint is not defined")

  # data
  if args.dataset == 'Deep360':  # deep 360
    test_left_img, test_right_img, test_left_disp = list_deep360_disparity_test(args.dataset_root, soiled=args.soiled)
    testDispData = Deep360DatasetDisparity(leftImgs=test_left_img, rightImgs=test_right_img, disps=test_left_disp)
    testDispDataLoader = torch.utils.data.DataLoader(testDispData, batch_size=args.batch_size, num_workers=args.batch_size, pin_memory=False, shuffle=False)

  # testing
  testDisp(model_disp, testDispDataLoader, args.checkpoint_disp, len(testDispData))


if __name__ == '__main__':
  main()