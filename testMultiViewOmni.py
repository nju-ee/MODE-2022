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
from torchsummary import summary
from tqdm import tqdm

from models import PSMstackhourglass, LCV360SD, AANet, ModesDisparity
from models import Fusion0, FusionUnet, FusionDilate, FusionAttentionUnet, FusionConf, FusionMultiScale, FusionDualBranch, FusionWithRGB, FusionSplitBranch
from utils.dispAndDepth import DispDepthTransformerCassini as DDTC
from utils.ERPandCassini import ERP2CA, CA2ERP
import utils.projection as projection
from utils.CassiniViewTrans import *
from utils.evalMetrics import defaultMetrics, compute_errors, getMetrics, create_image_grid

from dataloader.dataset3D60Loader import Dataset3D60
from dataloader.multi360DepthLoader import Multi360DepthDataset, Multi360FusionDataset, Multi360DepthSoiledDataset, Multi360FusionSoiledDataset
from dataloader.multiFisheyeLoader import OmniFisheyeDataset

parser = argparse.ArgumentParser(description='Multi View Omnidirectional Depth Estimation')

parser.add_argument("--dataset", default="deep360", choices=["deep360", "3d60", "OmniHouse", "Sunny", "Cloudy", "Sunset"], type=str, help="dataset to train on.")
parser.add_argument("--filelist", default="deep360", choices=["deep360", "suncg", "mat3d", "3d60", "panosuncg", "stanford2d3d", "matterport3d"], type=str, help="dataset name file.")
parser.add_argument("--dataset_root", default="../../datasets/Deep360/depth_on_1/", type=str, help="dataset root directory.")
parser.add_argument('--intermedia_path',
                    default='./outputs/depth_on_1_inter',
                    help='intermedia results saving path. directory to save predict depth maps transformed form disparity maps using for fusion')
parser.add_argument("--mode", default="disp", choices=["disp", "trans", "fusion", "intermedia_fusion"], type=str, help="dataset name file.")
parser.add_argument('--disp_model', default='Modes', help='select model')
parser.add_argument('--fusion_model', default='Fusion0', help='select fusion model')
parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--widthE', default=1024, type=int, help="width of omnidirectional images in ERP domain")
parser.add_argument('--heightE', default=512, type=int, help="height of omnidirectional images in ERP domain")
parser.add_argument('--widthC', default=512, type=int, help="width of omnidirectional images in Cassini domain")
parser.add_argument('--heightC', default=1024, type=int, help="height of omnidirectional images in Cassini domain")
parser.add_argument('--batch_size_disp', type=int, default=4, help='maxium disparity')
parser.add_argument('--batch_size_fusion', type=int, default=1, help='maxium disparity')
parser.add_argument('--max_depth', default=1000, type=float, help="max valid depth")
parser.add_argument('--checkpoint_disp', default='./checkpoints/pretrained_sceneflow.tar', help='load checkpoint of disparity estimation path')
parser.add_argument('--checkpoint_conf', default=None, help='load checkpoint of confidence estimation path')
parser.add_argument('--checkpoint_fusion', default='./checkpoints/ckpt_fusion_deep360_17.tar', help='load checkpoint of fusion module path')
parser.add_argument('--parallel_disp', action='store_true', default=False, help='test disp model parallel')
parser.add_argument('--parallel_fusion', action='store_true', default=False, help='test fusion model parallel')
parser.add_argument('--save_output_path', type=str, default=None, help='path to save output files. if set to None, will not save')
parser.add_argument('--save_ori', action='store_true', default=False, help='save original disparity or depth value map')
parser.add_argument('--summary', action='store_true', default=False, help='if set, will print the summary of models')
parser.add_argument("--view_trans", default="point_gather", choices=["grid_sample", "point_gather"], type=str, help="method for view transformation")
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--soiled', action='store_true', default=False, help='test soiled image')
parser.add_argument('--erp', action='store_true', default=False, help='test soiled image')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heightC, widthC = args.heightC, args.widthC
heightE, widthE = args.heightE, args.widthE
weights = torch.ones(1, 1, heightC, widthC).to(device)
weightsERP = torch.ones(1, 1, heightE, widthE).to(device)
sampling = torch.ones(1, 1, heightC, widthC).to(device)
dispDepthT = CassiniViewDepthTransformer()
save_out = args.save_output_path is not None
args.cuda = not args.no_cuda and torch.cuda.is_available()


def saveOutputOriValue(pred, gt, mask, rootDir, id, names=None, cons=True):
  b, c, h, w = pred.shape
  pred[~mask] = 0
  gt[~mask] = 0
  for i in range(b):
    predSave = pred[i, ::].cpu()
    gtSave = gt[i, ::].cpu()
    maskSave = mask[i, ::].cpu()
    saveimg = predSave.squeeze_(0).numpy()
    if names is None:
      prefix = "{:0>4}_test".format(id + i)
    else:
      oriName = names[i]
      if isinstance(oriName, list) or isinstance(oriName, tuple):
        oriName = oriName[0]
      oriName = oriName.replace(args.dataset_root, '')
      oriName = oriName.replace(args.intermedia_path, '')
      oriName = oriName.replace('../', '')
      oriName = oriName.replace('./', '')
      oriName = oriName.replace('/', '+')
      prefix = oriName.split('.')[0]
    # cv2.imwrite(os.path.join(rootDir, prefix + '.exr'), saveimg)
    np.save(os.path.join(rootDir, prefix + '.npy'), saveimg)


def saveOutput(pred, gt, mask, rootDir, id, names=None, log=True, cons=True, savewithGt=True):
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
    else:
      saveimg = predSave.squeeze_(0).numpy()
    saveimg = (saveimg - np.min(saveimg)) / (np.max(saveimg) - np.min(saveimg)) * 255

    saveimg = saveimg.astype(np.uint8)
    if names is None:
      prefix = "{:0>4}_test".format(id + i)
    else:
      oriName = names[i]
      if isinstance(oriName, list) or isinstance(oriName, tuple):
        oriName = oriName[0]
      oriName = oriName.replace(args.dataset_root, '')
      oriName = oriName.replace(args.intermedia_path, '')
      oriName = oriName.replace('../', '')
      oriName = oriName.replace('./', '')
      oriName = oriName.replace('/', '+')

      prefix = oriName.split('.')[0]
    saveimg = cv2.applyColorMap(saveimg, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(rootDir, prefix + '.png'), saveimg)


def testDisp(modelDisp, testDispDataLoader, modelNameDisp, numTestData):
  eType = 'disp'
  error_names = getMetrics(eType)
  if save_out:
    if not os.path.isdir(args.save_output_path):
      os.makedirs(args.save_output_path)
  modelDisp.eval()
  print("Testing of Disparity. Model: {}".format(modelNameDisp))
  print("num of test files: {}".format(numTestData))
  counter = 0
  errorsPred = np.zeros((len(error_names), numTestData), np.float32)
  with torch.no_grad():
    for batch_idx, batchData in enumerate(tqdm(testDispDataLoader, desc='Test iter')):
      leftImg = batchData['leftImg'].to(device)
      rightImg = batchData['rightImg'].to(device)
      dispMap = batchData['dispMap'].to(device)
      mask = (dispMap >= 0) & (~torch.isnan(dispMap)) & (~torch.isinf(dispMap)) & (dispMap < args.max_disp)
      b, c, h, w = leftImg.shape
      if args.dataset == "OmniHouse" or args.dataset == "Sunny" or args.dataset == "Cloudy" or args.dataset == "Sunset":
        inMask = batchData['invalidMask'].cuda()
        mask = mask & (~inMask)  # mask
      if args.disp_model == 'AANet':
        leftImg1 = F.interpolate(leftImg, size=(1080, 540))
        rightImg1 = F.interpolate(rightImg, size=(1080, 540))
        output = modelDisp(leftImg1, rightImg1)[-1]
        output = F.interpolate(output, size=(1024, 512))
        output = output / 540.0 * 512.0
      else:
        output = modelDisp(leftImg, rightImg)
      invalidMask = ~mask
      errors = compute_errors(dispMap, output, invalidMask, weights=weights, type=eType)
      for k in range(len(errors)):
        errorsPred[k, counter:counter + b] = errors[k].squeeze_(-1).squeeze_(-1).squeeze_(-1).cpu()
      if save_out:
        if args.save_ori: saveOutputOriValue(output, dispMap, mask, args.save_output_path, counter, names=batchData['leftNames'])  # save npy
        saveOutput(output, dispMap, mask, args.save_output_path, counter, names=batchData['leftNames'], log=True)
      counter += b
  mean_errorsPred = errorsPred.mean(1)
  np.savetxt('errorsPred.txt', errorsPred)
  print("Results of Disparity Estimation (test on: {}): ".format(modelNameDisp))
  print("\t|" + ("{:^10}|" * len(error_names)).format(*error_names))
  print("\t" + "-" * (len(error_names) * 11))
  print("\t|" + ("{:^10.4f}|" * len(mean_errorsPred)).format(*mean_errorsPred))


def testTransDepth(modelDisp, testFusionDataLoader, modelNameDisp, numTestData):
  eType = 'depth'
  error_names = getMetrics(eType)
  if save_out:
    if not os.path.isdir(args.save_output_path):
      os.makedirs(args.save_output_path)
  modelDisp.eval()
  print("Testing of Disparity. Model: {}".format(modelNameDisp))
  print("num of test files: {}".format(numTestData))
  counter = 0
  counterSave = 0
  errorsPredAll = np.zeros((len(error_names), numTestData * 6), np.float32)
  print(errorsPredAll.shape)
  errorsPredViews = []
  errorIdx = []
  camPairs = ['12', '13', '14', '23', '24', '34']
  ca2e = CA2ERP(args.heightE, args.widthE, args.heightC, args.widthC)
  for i in range(len(camPairs)):
    errorsPredViews.append(np.zeros((len(error_names), numTestData), np.float32))
    errorIdx.append(0)
    print(errorsPredViews[i].shape)
  sum = 0
  with torch.no_grad():
    for batch_idx, batchData in enumerate(tqdm(testFusionDataLoader, desc='Test iter')):
      imgPairs = batchData['imgPairs']
      depthGT = batchData['depthMap'].cuda()
      depthGTERP = ca2e.trans(depthGT, '0')
      depthGTERP[depthGTERP > args.max_depth] = args.max_depth
      mask = (depthGT > 0) & (~torch.isnan(depthGT)) & (~torch.isinf(depthGT)) & (depthGT <= args.max_depth)
      maskERP = (depthGTERP > 0) & (~torch.isnan(depthGTERP)) & (~torch.isinf(depthGTERP)) & (depthGTERP <= args.max_depth)
      #assert len(imgPairs) == 6
      for p in range(len(imgPairs)):
        leftImg, rightImg = imgPairs[p][0].cuda(), imgPairs[p][1].cuda()
        b, c, h, w = leftImg.shape
        output = modelDisp(leftImg, rightImg)
        if args.view_trans == 'grid_sample':
          predDep = dispDepthT.dispToDepthTrans(disp=output, camPair=p, maxDisp=args.max_disp, maxDepth=args.max_depth)
        elif args.view_trans == 'point_gather':
          predDep = batchViewTransOri(output, p, max_depth=args.max_depth)
        predDep[predDep > args.max_depth] = args.max_depth  #
        mask1 = (predDep > 0) & (~torch.isnan(depthGT)) & (~torch.isinf(depthGT))
        mask2 = mask * mask1
        invalidMask = ~mask2
        predDepERP = ca2e.trans(predDep, '0')
        predDepERP[predDepERP > args.max_depth] = args.max_depth
        if p < 6:
          errors = compute_errors(depthGT.clone(), predDep, invalidMask, weights=weights, type=eType)
          for k in range(len(errors)):
            errorsPredAll[k, counter:counter + b] = errors[k].squeeze_(-1).squeeze_(-1).squeeze_(-1).cpu()
          counter += b
          errors = compute_errors(depthGTERP.clone(), predDepERP, ~maskERP, weights=weightsERP, type=eType)
          for k in range(len(errors)):
            errorsPredViews[p][k, errorIdx[p]:errorIdx[p] + b] = errors[k].squeeze_(-1).squeeze_(-1).squeeze_(-1).cpu()
          errorIdx[p] += b
        if save_out:
          if args.save_ori: saveOutputOriValue(predDepERP, depthGTERP.clone(), maskERP, args.save_output_path, counter, names=batchData['leftNames'][p])
          saveOutput(predDepERP, depthGTERP.clone(), maskERP, args.save_output_path, counterSave, names=batchData['leftNames'][p], savewithGt=False)
        counterSave += b
  print(errorsPredAll.shape)
  mean_errorsPred = errorsPredAll.mean(1)
  print("Results of Disp-trans Depth Estimation (test on: {}): ".format(modelNameDisp))
  print("\t|" + ("{:^10}|" * len(error_names)).format(*error_names))
  print("\t" + "-" * (len(error_names) * 11))
  print("\t|" + ("{:^10.4f}|" * len(mean_errorsPred)).format(*mean_errorsPred))
  for i in range(len(camPairs)):
    print(errorsPredViews[i].shape)
    mean_errorsPred = errorsPredViews[i].mean(1)
    print("Results of Disp-trans depth Estimation (test on: {}), views: {}: ".format(args.disp_model, camPairs[i]))
    print("\t|" + ("{:^10}|" * len(error_names)).format(*error_names))
    print("\t" + "-" * (len(error_names) * 11))
    print("\t|" + ("{:^10.4f}|" * len(mean_errorsPred)).format(*mean_errorsPred))


def testFusionDepth(modelDisp, modelFusion, testFusionDataLoader, modelNameDisp, modelNameFusion, numTestData):
  eType = 'depth'
  error_names = getMetrics(eType)
  if save_out:
    if not os.path.isdir(args.save_output_path):
      os.makedirs(args.save_output_path)
  modelDisp.eval()
  modelFusion.eval()
  counter = 0
  errorsPred = np.zeros((len(error_names), numTestData), np.float32)
  print("Testing of Depth")
  print("num of test files: {}".format(numTestData))
  with torch.no_grad():
    for batch_idx, batchData in enumerate(tqdm(testFusionDataLoader, desc='Test iter')):
      imgPairs = batchData['imgPairs']
      depthGT = batchData['depthMap'].cuda()
      mask = (depthGT > 0) & (~torch.isnan(depthGT)) & (~torch.isinf(depthGT)) & (depthGT <= args.max_depth)
      assert len(imgPairs) == 6
      depthMaps = []
      for i in range(len(imgPairs)):
        leftImg, rightImg = imgPairs[i][0].cuda(), imgPairs[i][1].cuda()
        disp_pred = modelDisp(leftImg, rightImg).detach()
        if args.view_trans == 'grid_sample':
          depthMaps.append(dispDepthT.dispToDepthTrans(disp_pred, i, maxDisp=args.max_disp, maxDepth=args.max_depth))
        elif args.view_trans == 'point_gather':
          depthMaps.append(batchViewTransOri(disp_pred, i, max_depth=args.max_depth))
      depthMaps = torch.cat(depthMaps, dim=1)
      pred = modelFusion(depthMaps)
      invalidMask = ~mask
      b, c, h, w = depthMaps.shape
      errors = compute_errors(depthGT, pred, invalidMask, weights=weights, type=eType)
      for k in range(len(errors)):
        errorsPred[k, counter:counter + b] = errors[k].squeeze_(-1).squeeze_(-1).squeeze_(-1).cpu()
      if save_out:
        saveOutput(pred, depthGT, mask, args.save_output_path, counter, names=batchData['depthName'])
      counter += b
  mean_errorsPred = errorsPred.mean(1)
  print("Results of Fusion Depth Estimation (test on: {} - {}): ".format(modelNameDisp, modelNameFusion))
  print("\t|" + ("{:^10}|" * len(error_names)).format(*error_names))
  print("\t" + "-" * (len(error_names) * 11))
  print("\t|" + ("{:^10.4f}|" * len(mean_errorsPred)).format(*mean_errorsPred))


def testFusionIntermediaDepth(modelFusion, testFusionDataLoader, modelNameDisp, modelNameFusion, numTestData, calERP=True):
  eType = 'depth'
  error_names = getMetrics(eType)
  if save_out:
    if not os.path.isdir(args.save_output_path):
      os.makedirs(args.save_output_path)
      if calERP:
        os.makedirs(args.save_output_path + '/erp')
  modelFusion.eval()
  counter = 0
  errorsPred = np.zeros((len(error_names), numTestData), np.float32)
  errorsPredERP = np.zeros((len(error_names), numTestData), np.float32)
  ca2e = CA2ERP(args.heightE, args.widthE, args.heightC, args.widthC)
  weightsERP = torch.ones(1, 1, args.heightE, args.widthE).cuda()
  print("Testing of Depth")
  print("num of test files: {}".format(numTestData))
  with torch.no_grad():
    for batch_idx, batchData in enumerate(tqdm(testFusionDataLoader, desc='Test iter')):
      inputs = batchData['inputs'].cuda()
      depthGT = batchData['depthMap'].cuda()
      mask = (depthGT > 0) & (~torch.isnan(depthGT)) & (~torch.isinf(depthGT)) & (depthGT <= args.max_depth)
      if args.fusion_model == 'withRGB':
        rgbImgs = batchData['rgbImgs'].cuda()
        pred = modelFusion(inputs, rgbImgs)
      else:
        pred = modelFusion(inputs)
      invalidMask = ~mask
      b, c, h, w = pred.shape
      #pred[pred < 1e-10] = 1e-10
      # abs_rel_t, sq_rel_t, rmse_t, rmse_log_t, mae_t, d1, sie, px3, px1, delta1, delta2, delta3 = compute_errors(depthGT, pred, invalidMask, weights=weights)
      errors = compute_errors(depthGT, pred, invalidMask, weights=weights, type=eType)
      predERP = ca2e.trans(pred, '0')
      depthERP = ca2e.trans(depthGT, '0')
      depthERP[depthERP > args.max_depth] = args.max_depth
      maskERP = (depthERP > 0) & (~torch.isnan(depthERP)) & (~torch.isinf(depthERP)) & (depthERP <= args.max_depth)
      invalidMaskERP = ~maskERP
      errorsERP = compute_errors(depthERP, predERP, invalidMaskERP, weights=weightsERP, type=eType)

      # errorsPred[:, idx] = abs_rel_t[i], sq_rel_t[i], rmse_t[i], rmse_log_t[i], mae_t[i], d1[i], sie[i], px3[i], px1[i], delta1[i], delta2[i], delta3[i]
      for k in range(len(errors)):
        errorsPred[k, counter:counter + b] = errors[k].squeeze_(-1).squeeze_(-1).squeeze_(-1).cpu()
        errorsPredERP[k, counter:counter + b] = errorsERP[k].squeeze_(-1).squeeze_(-1).squeeze_(-1).cpu()
        #print(errorsPred[k, counter:counter + b])
      if save_out:
        saveOutput(pred, depthGT, mask, args.save_output_path, counter, names=batchData['depthName'])
        saveOutputOriValue(predERP, depthERP, maskERP, args.save_output_path + '/erp', counter, names=batchData['depthName'])
        saveOutput(predERP, depthERP, maskERP, args.save_output_path + '/erp', counter, names=batchData['depthName'])

      counter += b
  mean_errorsPred = errorsPred.mean(1)
  mean_errorsPredERP = errorsPredERP.mean(1)
  print("Results of Fusion Depth Estimation (test on: {} - {}): ".format(modelNameDisp, modelNameFusion))
  print("\t|" + ("{:^10}|" * len(error_names)).format(*error_names))
  print("\t" + "-" * (len(error_names) * 11))
  print("\t|" + ("{:^10.4f}|" * len(mean_errorsPred)).format(*mean_errorsPred))
  print("Results of Fusion Depth Estimation (test on: {} - {}) ERP: ".format(modelNameDisp, modelNameFusion))
  print("\t|" + ("{:^10}|" * len(error_names)).format(*error_names))
  print("\t" + "-" * (len(error_names) * 11))
  print("\t|" + ("{:^10.4f}|" * len(mean_errorsPredERP)).format(*mean_errorsPredERP))


def main():
  # model
  dispModelDict = {'360SDNet': LCV360SD, 'PSMNet': PSMstackhourglass, 'AANet': AANet, 'Modes': ModesDisparity}
  fusionModelDict = {
      'Fusion0': Fusion0,
      'Unet': FusionUnet,
      'Dilate': FusionDilate,
      'AttUnet': FusionAttentionUnet,
      'Conf': FusionConf,
      'MultiScale': FusionMultiScale,
      'Dual': FusionDualBranch,
      'withRGB': FusionWithRGB,
      'Split': FusionSplitBranch
  }

  if args.mode != 'intermedia_fusion':
    model_Disp = dispModelDict[args.disp_model](args.max_disp)
  else:
    model_Disp = None

  if args.mode == 'fusion' or args.mode == 'intermedia_fusion':
    modelFusion = fusionModelDict[args.fusion_model](max_depth=args.max_depth)
  else:
    modelFusion = None

  if (model_Disp is not None):
    if (args.parallel_disp):
      model_Disp = nn.DataParallel(model_Disp)
    if args.cuda:
      model_Disp.cuda()

  if (modelFusion is not None):
    if (args.parallel_fusion):
      modelFusion = nn.DataParallel(modelFusion)
    if args.cuda:
      modelFusion.cuda()

  if (model_Disp is not None):
    if (args.checkpoint_disp is not None):
      state_dict = torch.load(args.checkpoint_disp)
      model_Disp.load_state_dict(state_dict['state_dict'])
    else:
      raise ValueError("disp model checkpoint is not defined")

  if (modelFusion is not None):
    if (args.checkpoint_fusion is not None):
      state_dict = torch.load(args.checkpoint_fusion)
      modelFusion.load_state_dict(state_dict['state_dict'])
    else:
      raise ValueError("fusion model checkpoint is not defined")

  #summary
  if args.summary:
    try:
      print(model_Disp)
      # writer = SummaryWriter('./logs')
      # with writer:
      #   dummyL = torch.randn(1, 3, args.heightC, args.widthC)
      #   dummyR = torch.randn(1, 3, args.heightC, args.widthC)
      #   print(dummyL.shape)
      #   dummyD = torch.randn(1, 6, args.heightC, args.widthC)
      #   if args.mode == 'disp':
      #     writer.add_graph(model_Disp, (dummyL, dummyR))
      #   elif args.mode == 'intermedia_fusion':
      #     writer.add_graph(modelFusion, dummyD)
      summary(model_Disp, [(3, heightC, widthC), (3, heightC, widthC)])
      if args.mode == 'fusion':
        summary(modelFusion, (6, heightC, widthC))
    except Exception as e:
      print("summary failed. {}".format(e))
  catEquiInfo = True if args.disp_model == '360SDNet' else False

  # data
  if args.dataset == 'deep360':  # deep 360
    myDataset = Multi360DepthSoiledDataset if args.soiled else Multi360DepthDataset
    if args.mode == 'disp':
      testDispData = myDataset(dataUsage='disparity', crop=False, rootDir=args.dataset_root, curStage='testing', catEquiInfo=catEquiInfo)
      testDispDataLoader = torch.utils.data.DataLoader(testDispData, batch_size=args.batch_size_disp, num_workers=4, pin_memory=False, shuffle=False)
    elif args.mode == 'intermedia_fusion':
      myDataset = Multi360FusionSoiledDataset if args.soiled else Multi360FusionDataset
      testFusionData = myDataset(rootInputDir=args.intermedia_path, rootDepthDir=args.dataset_root, curStage='testing')
      testFusionDataLoader = torch.utils.data.DataLoader(testFusionData, batch_size=args.batch_size_fusion, num_workers=4, pin_memory=False, shuffle=False)
    elif args.mode == 'trans' or args.mode == 'fusion':
      testFusionData = myDataset(dataUsage='fusion', crop=False, rootDir=args.dataset_root, curStage='testing', catEquiInfo=catEquiInfo)
      testFusionDataLoader = torch.utils.data.DataLoader(testFusionData, batch_size=args.batch_size_fusion, num_workers=4, pin_memory=False, shuffle=False)
    else:
      raise NotImplementedError("mode <{}> is not supported!".format(args.mode))
  # multi fisheye datasets
  elif args.dataset == "OmniHouse" or args.dataset == "Sunny" or args.dataset == "Cloudy" or args.dataset == "Sunset":
    myDataset = OmniFisheyeDataset
    if args.mode == 'disp':
      testDispData = myDataset(mode='disparity',
                               curStage='testing',
                               datasetName=args.dataset,
                               shape=(640,
                                      320),
                               crop=False,
                               catEquiInfo=False,
                               soiled=False,
                               shuffleOrder=False,
                               inputRGB=True,
                               needMask=False)
      testDispDataLoader = torch.utils.data.DataLoader(testDispData, batch_size=args.batch_size_disp, num_workers=4, pin_memory=False, shuffle=False)
    elif args.mode == 'intermedia_fusion':
      testFusionData = myDataset(mode='intermedia_fusion',
                                 curStage='testing',
                                 datasetName=args.dataset,
                                 shape=(640,
                                        320),
                                 crop=False,
                                 catEquiInfo=False,
                                 soiled=False,
                                 shuffleOrder=False,
                                 inputRGB=True,
                                 needMask=False)
      testFusionDataLoader = torch.utils.data.DataLoader(testFusionData, batch_size=args.batch_size_fusion, num_workers=4, pin_memory=False, shuffle=False)
    elif args.mode == 'trans' or args.mode == 'fusion':
      testFusionData = myDataset(mode='fusion',
                                 curStage='testing',
                                 datasetName=args.dataset,
                                 shape=(640,
                                        320),
                                 crop=False,
                                 catEquiInfo=False,
                                 soiled=False,
                                 shuffleOrder=False,
                                 inputRGB=True,
                                 needMask=False)
      testFusionDataLoader = torch.utils.data.DataLoader(testFusionData, batch_size=args.batch_size_fusion, num_workers=4, pin_memory=False, shuffle=False)
    else:
      raise NotImplementedError("mode <{}> is not supported!".format(args.mode))
  # others
  elif args.dataset == '3d60':
    if args.mode == 'all' or args.mode == 'disp':
      testDispData = Dataset3D60(filenamesFile='./dataloader/3d60_test.txt',
                                 rootDir=args.dataset_root,
                                 mode='disparity',
                                 curStage='testing',
                                 shape=(args.heightC,
                                        args.widthC),
                                 crop=False,
                                 catEquiInfo=False,
                                 soiled=False,
                                 shuffleOrder=False,
                                 inputRGB=True,
                                 needMask=False)
      testDispDataLoader = torch.utils.data.DataLoader(testDispData, batch_size=args.batch_size_disp, num_workers=4, pin_memory=False, shuffle=False)
    elif args.mode == "intermedia_fusion" or args.mode == "fusion" or args.mode == "trans":
      testFusionData = Dataset3D60(filenamesFile='./dataloader/3d60_test.txt',
                                   rootDir=args.dataset_root,
                                   interDir=args.intermedia_path,
                                   mode='fusion',
                                   curStage='testing',
                                   shape=(args.heightC,
                                          args.widthC),
                                   crop=False,
                                   catEquiInfo=False,
                                   soiled=False,
                                   shuffleOrder=False,
                                   inputRGB=True,
                                   needMask=False)
      testFusionDataLoader = torch.utils.data.DataLoader(testFusionData, batch_size=args.batch_size_fusion, num_workers=4, pin_memory=False, shuffle=False)
  else:
    raise NotImplementedError("Dataset <{}> is not supported yet!".format(args.dataset))

  # testing
  if args.mode == 'disp':
    testDisp(model_Disp, testDispDataLoader, args.checkpoint_disp, len(testDispData))
  else:
    if args.mode == 'trans':
      testTransDepth(model_Disp, testFusionDataLoader, args.checkpoint_disp, len(testFusionData))
    elif args.mode == 'fusion':
      testFusionDepth(model_Disp, modelFusion, testFusionDataLoader, args.checkpoint_disp, args.checkpoint_fusion, len(testFusionData))
    elif args.mode == 'intermedia_fusion':
      testFusionIntermediaDepth(modelFusion, testFusionDataLoader, args.checkpoint_disp, args.checkpoint_fusion, len(testFusionData))
    else:
      raise NotImplementedError("mode <{}> is not supported!".format(args.mode))


if __name__ == '__main__':
  main()