from __future__ import print_function
import os
import sys
sys.path.append('./')
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

from models import PSMstackhourglass, LCV360SD, AANet, ModesDisparity
from models import Fusion0, FusionUnet, FusionDilate, FusionAttentionUnet, FusionConf, FusionMultiScale, FusionDualBranch, FusionWithRGB, FusionSplitBranch
from utils.dispAndDepth import DispDepthTransformerCassini as DDTC
import utils.ERPandCassini as ErpCa
import utils.projection as projection
from utils.CassiniViewTrans import *
from utils.loss import scaleInvariantLoss, psmnetSHGLoss, aanetMultiScaleLoss, MultiscaleInvariantLoss, BerHuLoss, MultiscaleBerHuLoss, scaleInvariantLossWithSoilMask
from utils.evalMetrics import getMetrics, compute_errors
from dataloader.dataset3D60Loader import Dataset3D60
from dataloader.multi360DepthLoader import Multi360DepthDataset, Multi360FusionDataset, Multi360DepthSoiledDataset, Multi360FusionSoiledDataset, Multi360FusionOneFrameDataset
from dataloader.multiFisheyeLoader import OmniFisheyeDataset
from models import initModel, loadPartialModel, freezeSHG, unfreezeSHG, loadPartialModel360SD
'''
Argument Definition
'''

parser = argparse.ArgumentParser(description='Multi View Omnidirectional Depth Estimation')

# model
parser.add_argument('--disp_model', default='Modes', help='select model')
parser.add_argument('--fusion_model', default='Fusion0', help='select fusion model')
# data
parser.add_argument("--dataset", default="deep360", choices=["deep360", "3d60", "OmniHouse", "Sunny", "Cloudy", "Sunset"], type=str, help="dataset to train on.")
parser.add_argument("--dataset_root", default="../../datasets/Deep360/depth_on_1/", type=str, help="dataset root directory.")
parser.add_argument("--filelist", default="deep360", choices=["deep360", "suncg", "mat3d", "3d60", "panosuncg", "stanford2d3d", "matterport3d"], type=str, help="dataset name file.")
parser.add_argument('--intermedia_path',
                    default='./outputs/depth_on_1_inter',
                    help='intermedia results saving path. directory to save predict depth maps transformed form disparity maps using for fusion')
parser.add_argument('--widthE', default=1024, type=int, help="width of omnidirectional images in ERP domain")
parser.add_argument('--heightE', default=512, type=int, help="height of omnidirectional images in ERP domain")
parser.add_argument('--widthC', default=512, type=int, help="width of omnidirectional images in Cassini domain")
parser.add_argument('--heightC', default=1024, type=int, help="height of omnidirectional images in Cassini domain")
# stereo
parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--max_depth', default=1000, type=float, help="max valid depth")
parser.add_argument('--baseline', default=1, type=float, help="baseline of binocular spherical system")

# hyper parameters
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--start_decay', type=int, default=15, help='number of epoch for lr to start decay')
parser.add_argument('--start_learn', type=int, default=10, help='number of epoch for LCV to start learn')
parser.add_argument('--batch_size_disp', type=int, default=8, help='number of batch to train')
parser.add_argument('--batch_size_fusion', type=int, default=4, help='number of batch to train')
parser.add_argument('--learning_rate_disp', type=float, default=0.001, help='learning rate of disp estimation training')
parser.add_argument('--learning_rate_fusion', type=float, default=0.001, help='learning rate of fusion estimation training')

# trainging
parser.add_argument('--mode', type=str, default="all", choices=["all", "disp", "confidence", "fusion", "intermedia_fusion", "refine"], help='load checkpoint and resume learning')
parser.add_argument('--resume', action='store_true', default=False, help='load checkpoint and resume learning')
parser.add_argument('--checkpoint_disp', default=None, help='load checkpoint of disparity estimation path')
parser.add_argument('--loadSHG', action='store_true', default=False, help='if set,load stack hour glass part from pretrained PSMNet, while skip feature extraction part')
parser.add_argument('--freezeSHG', action='store_true', default=False, help='if set,freeze stack hour glass part in first several epoch, and unfreeze from the epoch identyfied by arg start_learn')
parser.add_argument('--checkpoint_conf', default=None, help='load checkpoint of confidence estimation path')
parser.add_argument('--checkpoint_fusion', default=None, help='load checkpoint of fusion module path')
parser.add_argument('--pretrained', default=None, help='load pretrained disp model path')
parser.add_argument('--tensorboard_path', default='./logs', help='tensorboard path')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--printNet', action='store_true', default=False, help='print network details')
parser.add_argument('--crop_disp', action='store_true', default=False, help='crop when train disp')
parser.add_argument('--parallel', action='store_true', default=False, help='train model parallel')
parser.add_argument('--cudnn_deter', action='store_true', default=False, help='if True, set cudnn deterministic as True and benchmark as False. Otherwise the opposite')
parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 123)')
parser.add_argument("--view_trans", default="point_gather", choices=["grid_sample", "point_gather"], type=str, help="method for view transformation")
parser.add_argument('--fusion_shuffle_order', action='store_true', default=False, help='shuffle input order in fusion')
parser.add_argument("--scheduler", action='store_true', default=False, help="method for view transformation")
parser.add_argument('--soiled', action='store_true', default=False, help='test soiled image')
parser.add_argument('--soil_mask_loss', action='store_true', default=False, help='use soil mask weighted loss')
# saving
parser.add_argument('--save_suffix_disp', type=str, default=None, help='save checkpoint name')
parser.add_argument('--save_suffix_fusion', type=str, default=None, help='save checkpoint name')
parser.add_argument('--save_checkpoint_path', default='./checkpoints', help='save checkpoint path')
parser.add_argument('--save_image_path', type=str, default='./outputs', help='save images path')

args = parser.parse_args()

print("Training!")
print("Args:\n{}".format(args))
# cuda & device
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------------------------
# Random Seed -----------------------------
torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# cudnn benchmark and deterministric
torch.backends.cudnn.benchmark = not args.cudnn_deter
torch.backends.cudnn.deterministic = args.cudnn_deter
# ------------------------------------------
'''
Functions
'''


# Freeze / Unfreeze Function
# freeze_layer ----------------------
def freeze_layer(layer):
  for param in layer.parameters():
    param.requires_grad = False


# Unfreeze_layer --------------------
def unfreeze_layer(layer):
  for param in layer.parameters():
    param.requires_grad = True


# -----------------------------------------------------------------------------


# Save / Load Checkpoints Functions
def saveCkpt(epoch, avgLoss, model, stage, model_name):
  savefilename = args.save_checkpoint_path + '/ckpt_' + str(stage) + '_' + str(model_name)
  if stage == 'disp' and (args.save_suffix_disp is not None):
    savefilename = savefilename + '_' + args.save_suffix_disp
  elif stage == 'fusion' and (args.save_suffix_fusion is not None):
    savefilename = savefilename + '_' + args.save_suffix_fusion
  savefilename = savefilename + '_' + str(epoch) + '.tar'
  torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'train_loss': avgLoss}, savefilename)


def loadCkpt(model, checkpoint):
  if checkpoint is not None:
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict['state_dict'])
    start_epoch = state_dict['epoch']
  else:
    print('checkpoint is None, start training form epoch 0')
  print('checkpoint Name: {}, Number of model parameters: {}'.format(checkpoint, sum([p.data.nelement() for p in model.parameters()])))
  return model, start_epoch


# ------------------------------------


# Save validation output sample Function
def saveValOutputSample(val_output, mask, dispGt, e):
  b, c, h, w = dispGt.size()
  div = torch.ones([c, h, 10])
  gt = dispGt[0, ::].cpu()
  pred = val_output[0, ::].cpu()
  mask = mask[0, ::].cpu()
  # gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
  # pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred))
  div = torch.log10(div * 1000 + 1.0)
  gt[mask] = torch.log10(gt[mask] + 1.0)
  pred[mask] = torch.log10(pred[mask] + 1.0)
  gt[~mask] = 0
  pred[~mask] = 0
  saveimg = torch.cat([gt, div, pred], dim=2).squeeze_(0).numpy()
  saveimg = (saveimg - np.min(saveimg)) / (np.max(saveimg) - np.min(saveimg)) * 255
  saveimg = saveimg.astype(np.uint8)
  prefix = "{:0>3}_val".format(e)
  saveimg = cv2.applyColorMap(saveimg, cv2.COLORMAP_JET)
  # torchvision.utils.save_image(saveimg, os.path.join(imagePath, prefix + '.png'))
  cv2.imwrite(os.path.join(imagePath, prefix + '.png'), saveimg)


# --------------------------------------------------


# Adjust Learning Rate
def adjust_learning_rate(optimizer, epoch, learningRate):
  lr = learningRate
  if epoch > args.start_decay:
    lr = learningRate * 0.1
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


# Train and Valid Functions
def trainDisp(imgL, imgR, disp_true, mask, disp_model, disp_optimizer):
  disp_model.train()
  disp_optimizer.zero_grad()
  # Loss --------------------------------------------
  # output1, output2, output3 = disp_model(imgL, imgR)
  # loss = 0.5 * F.smooth_l1_loss(output1[mask],
  #                               disp_true[mask],
  #                               size_average=True) + 0.7 * F.smooth_l1_loss(output2[mask],
  #                                                                           disp_true[mask],
  #                                                                           size_average=True) + F.smooth_l1_loss(output3[mask],
  #                                                                                                                 disp_true[mask],
  #                                                                                                                 size_average=True)
  if args.disp_model == '360SDNet' or args.disp_model == 'PSMNet' or args.disp_model == 'Modes':
    outputs = list(disp_model(imgL, imgR))
    assert len(outputs) == 3
    loss = psmnetSHGLoss(outputs, disp_true, mask)
  elif args.disp_model == 'AANet':
    outputs = disp_model(imgL, imgR)
    h, w = imgL.shape[-2:]
    loss = aanetMultiScaleLoss(outputs, disp_true, mask, h, w)
  else:
    raise NotImplementedError("disp model not implemented error!!!")
  # --------------------------------------------------
  loss.backward()
  disp_optimizer.step()

  return loss.data.item()


def valDisp(imgL, imgR, disp_true, mask, disp_model):
  disp_model.eval()

  with torch.no_grad():
    if args.disp_model == '360SDNet' or args.disp_model == 'PSMNet' or args.disp_model == 'Modes':
      output = disp_model(imgL, imgR)
    elif args.disp_model == 'AANet':
      imgL_r = F.interpolate(imgL, size=(576, 288))
      imgR_r = F.interpolate(imgR, size=(576, 288))
      output = disp_model(imgL_r, imgR_r)[-1]
      output = F.interpolate(output, size=(args.heightC, args.widthC))
  if len(disp_true[mask]) == 0:
    loss = 0
  else:
    loss = torch.mean(torch.abs(output[mask] - disp_true[mask]))  # end-point-error

  return loss, output


def trainFusion(images, depth_true, mask, disp_model, fusion_model, optimizer_fusion):
  assert len(images) == 6
  disp_model.eval()
  fusion_model.train()
  optimizer_fusion.zero_grad()
  depthMaps = []
  for i in range(len(images)):
    leftImg, rightImg = images[i][0], images[i][1]
    disp_pred = disp_model(leftImg, rightImg).detach()
    if args.view_trans == 'grid_sample':
      depthMaps.append(dispDepthT.dispToDepthTrans(disp_pred, i, maxDisp=args.max_disp, maxDepth=args.max_depth))
    elif args.view_trans == 'point_gather':
      depthMaps.append(batchDisp2Depth(disp_pred, i))
  depthMaps = torch.cat(depthMaps, dim=1)
  pred = fusion_model(depthMaps)
  loss = F.smooth_l1_loss(pred[mask], depth_true[mask])
  loss.backward()
  optimizer_fusion.step()
  return loss.data.item()


def valFusion(images, depth_true, mask, disp_model, fusion_model):
  assert len(images) == 6
  disp_model.eval()
  fusion_model.eval()
  with torch.no_grad():
    depthMaps = []
    for i in range(len(images)):
      leftImg, rightImg = images[i][0].cuda(), images[i][1].cuda()
      disp_pred = disp_model(leftImg, rightImg).detach()
      if args.view_trans == 'grid_sample':
        depthMaps.append(dispDepthT.dispToDepthTrans(disp_pred, i, maxDisp=args.max_disp, maxDepth=args.max_depth))
      elif args.view_trans == 'point_gather':
        depthMaps.append(batchViewTransOri(disp_pred, i))
    depthMaps = torch.cat(depthMaps, dim=1)
    pred = fusion_model(depthMaps)
  if len(depth_true[mask]) == 0:
    loss = 0
  else:
    loss = torch.mean(torch.abs(pred[mask] - depth_true[mask]))
  return loss, pred


# train fusion from saved intermedia depth maps
def trainFusionIntermedia(inputs, depth_true, rgbImgs, mask, fusion_model, optimizer_fusion, soilMask=None):
  fusion_model.train()
  optimizer_fusion.zero_grad()
  if args.fusion_model == 'MultiScale' or args.fusion_model == 'Dual':
    pred = list(fusion_model(inputs))
    # loss = MultiscaleInvariantLoss(pred, depth_true, mask, lam=0.5)
    loss = MultiscaleBerHuLoss(pred, depth_true, mask)
  elif args.fusion_model == 'withRGB':
    pred = fusion_model(inputs, rgbImgs)
    #loss = scaleInvariantLoss(pred, depth_true, mask, lam=0.5) + 0.1 * BerHuLoss(pred, depth_true, mask)
    if args.soil_mask_loss:
      loss = scaleInvariantLossWithSoilMask(pred, depth_true, mask, soilMask, lam=0.5, rate=3)
    else:
      loss = scaleInvariantLoss(pred, depth_true, mask, lam=0.5)
  else:
    pred = fusion_model(inputs)
    # loss = F.smooth_l1_loss(pred[mask], depth_true[mask])
    # loss = 10 * scaleInvariantLoss(pred, depth_true, mask, lam=0.5) + 0.1 * F.smooth_l1_loss(pred[mask], depth_true[mask])
    loss = scaleInvariantLoss(pred, depth_true, mask, lam=0.5)
    #loss = scaleInvariantLoss(pred, depth_true, mask, lam=0.5) + 0.1 * BerHuLoss(pred, depth_true, mask)
  loss.backward()
  optimizer_fusion.step()
  return loss.data.item()


def valFusionIntermedia(inputs, depth_true, rgbImgs, mask, fusion_model):
  fusion_model.eval()
  with torch.no_grad():
    if args.fusion_model == 'withRGB':
      pred = fusion_model(inputs, rgbImgs)
    else:
      pred = fusion_model(inputs)
  if len(depth_true[mask]) == 0:
    loss = 0
  else:
    loss = torch.mean(torch.abs(pred[mask] - depth_true[mask]))
  return loss, pred


def runDispOneEpoch(trainDispDataLoader, valDispDataLoader, disp_model, optimizer, scheduler_disp, epoch, global_step, global_val):
  startTime = time.time()
  total_train_loss = 0
  if scheduler_disp is None:  # if no scheduler, than use manual adjust
    adjust_learning_rate(optimizer, epoch, args.learning_rate_disp)
  print("Epoch: {}, Current Stage: Disp, Current Learning Rate: {}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
  # -------------------------------
  # Train ----------------------------------
  for batch_idx, batchData in enumerate(tqdm(trainDispDataLoader, desc='Train iter {}'.format(epoch))):
    leftImg = batchData['leftImg'].cuda()
    rightImg = batchData['rightImg'].cuda()
    dispMap = batchData['dispMap'].cuda()
    # mask = (dispMap > 0) & (~torch.isnan(dispMap)) & (~torch.isinf(dispMap)) & (dispMap <= args.max_disp)
    mask = (~torch.isnan(dispMap))
    # for fish eye datasets
    b, c, h, w = leftImg.shape
    if args.dataset == "OmniHouse" or args.dataset == "Sunny" or args.dataset == "Cloudy" or args.dataset == "Sunset":
      invalidMask = batchData['invalidMask'].cuda()
      mask = (~torch.isnan(dispMap)) & (~invalidMask)
    loss = trainDisp(leftImg, rightImg, dispMap, mask, disp_model, optimizer)
    total_train_loss += loss
    global_step += 1
    writer.add_scalar('loss disp', loss, global_step)  # tensorboardX for iter
  writer.add_scalar('total disp train loss', total_train_loss / len(trainDispDataLoader), epoch)  # tensorboardX for epoch
  print("epoch: {}, avg train loss: {}".format(epoch, total_train_loss / len(trainDispDataLoader)))
  # ----------------------------------------------------

  # Save Checkpoint ------------------------------------
  saveCkpt(epoch, total_train_loss / len(trainDispDataLoader), disp_model, stage='disp', model_name=args.disp_model)
  # --------------------------------------------------------

  # Valid --------------------------------------------------
  eType = 'disp'
  error_names = getMetrics(eType)
  numTestData = args.batch_size_disp * len(valDispDataLoader)
  errorsPred = np.zeros((len(error_names), numTestData), np.float32)
  weights = torch.ones(1, 1, heightC, widthC).to(device)
  counter = 0
  total_val_loss = 0
  total_val_crop_rmse = 0
  for batch_idx, batchData in enumerate(tqdm(valDispDataLoader, desc='Train iter {}'.format(epoch))):
    leftImg = batchData['leftImg'].cuda()
    rightImg = batchData['rightImg'].cuda()
    dispMap = batchData['dispMap'].cuda()
    mask = (dispMap > 0) & (~torch.isnan(dispMap)) & (~torch.isinf(dispMap)) & (dispMap <= args.max_disp)
    b, c, h, w = leftImg.shape
    # for fish eye datasets
    if args.dataset == "OmniHouse" or args.dataset == "Sunny" or args.dataset == "Cloudy" or args.dataset == "Sunset":
      invalidMask = batchData['invalidMask'].cuda()
      mask2 = mask & (~invalidMask)
    else:
      mask2 = mask
    val_loss, val_output = valDisp(leftImg, rightImg, dispMap, mask2, disp_model)
    if batch_idx == 0:
      saveValOutputSample(val_output, mask2, dispMap, epoch)
    errors = compute_errors(dispMap.clone(), val_output, ~mask2, weights=weights, type=eType)
    for k in range(len(errors)):
      errorsPred[k, counter:counter + b] = errors[k].squeeze_(-1).squeeze_(-1).squeeze_(-1).cpu()
    counter += b
    val_crop_rmse = torch.sqrt(torch.mean((dispMap[mask2] - val_output[mask2])**2))
    # -------------------------------------------------------------
    # Loss ---------------------------------
    total_val_loss += val_loss
    total_val_crop_rmse += val_crop_rmse
    # ---------------------------------------
    # Step ------
    global_val += 1
    # ------------
  if scheduler_disp is not None:
    scheduler_disp.step()
  writer.add_scalar('total disp validation loss', total_val_loss / (len(valDispDataLoader)), epoch)  # tensorboardX for validation in epoch
  writer.add_scalar('total disp validation rmse', total_val_crop_rmse / (len(valDispDataLoader)), epoch)  # tensorboardX rmse for validation in epoch
  print("epoch: {}, avg val loss: {}, avg val rmse {}".format(epoch, total_val_loss / len(valDispDataLoader), total_val_crop_rmse / (len(valDispDataLoader))))
  mean_errorsPred = errorsPred.mean(1)
  print("validation results (test on epoch: {}, model <{}>): ".format(epoch, args.disp_model))
  print("\t|" + ("{:^10}|" * len(error_names)).format(*error_names))
  print("\t" + "-" * (len(error_names) * 11))
  print("\t|" + ("{:^10.4f}|" * len(mean_errorsPred)).format(*mean_errorsPred))
  print("Time of This epoch: {} seconds".format(time.time() - startTime))
  return global_step, global_val


def runFusionOneEpoch(trainFusionDataLoader, valFusionDataLoader, disp_model, fusion_model, optimizer, epoch, global_step, global_val):
  startTime = time.time()
  total_train_loss = 0
  adjust_learning_rate(optimizer, epoch, args.learning_rate_fusion)
  print("Epoch: {}, Current Stage: Fusion, Current Learning Rate: {}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
  # -------------------------------
  # Train ----------------------------------
  for batch_idx, batchData in enumerate(tqdm(trainFusionDataLoader, desc='Train iter {}'.format(epoch))):
    imgPairs = batchData['imgPairs']
    depthMap = batchData['depthMap'].cuda()
    mask = (depthMap > 0) & (~torch.isnan(depthMap)) & (~torch.isinf(depthMap)) & (depthMap <= args.max_depth)
    loss = trainFusion(imgPairs, depthMap, mask, disp_model, fusion_model, optimizer)
    total_train_loss += loss
    global_step += 1
    writer.add_scalar('loss fusion', loss, global_step)  # tensorboardX for iter
  writer.add_scalar('total fusion train loss', total_train_loss / len(trainFusionDataLoader), epoch)  # tensorboardX for epoch
  print("epoch: {}, avg train loss: {}".format(epoch, total_train_loss / len(trainDispDataLoader)))
  # ----------------------------------------------------

  # Save Checkpoint ------------------------------------
  saveCkpt(epoch, total_train_loss / len(trainFusionDataLoader), fusion_model, stage='fusion', model_name=args.fusion_model)
  # --------------------------------------------------------

  # Valid --------------------------------------------------
  total_val_loss = 0
  total_val_crop_rmse = 0
  for batch_idx, batchData in enumerate(tqdm(valFusionDataLoader, desc='Train iter {}'.format(epoch))):
    imgPairs = batchData['imgPairs']
    depthGT = batchData['depthMap'].cuda()
    mask = (depthGT > 0) & (~torch.isnan(depthGT)) & (~torch.isinf(depthGT)) & (depthGT <= args.max_depth)
    val_loss, val_output = valFusion(imgPairs, depthGT, mask, disp_model, fusion_model)
    if batch_idx == 0:
      saveValOutputSample(val_output, mask, depthGT, epoch)
    val_crop_rmse = torch.sqrt(torch.mean((depthGT[mask] - val_output[mask])**2))
    # -------------------------------------------------------------
    # Loss ---------------------------------
    total_val_loss += val_loss
    total_val_crop_rmse += val_crop_rmse
    # ---------------------------------------
    # Step ------
    global_val += 1
    # ------------
  writer.add_scalar('total fusion validation loss', total_val_loss / (len(valFusionDataLoader)), epoch)  # tensorboardX for validation in epoch
  writer.add_scalar('total fusion validation rmse', total_val_crop_rmse / (len(valFusionDataLoader)), epoch)  # tensorboardX rmse for validation in epoch
  print("epoch: {}, avg val loss: {}, avg val rmse {}".format(epoch, total_val_loss / len(valFusionDataLoader), total_val_crop_rmse / (len(valFusionDataLoader))))
  print("Time of This epoch: {} seconds".format(time.time() - startTime))
  return global_step, global_val


# train fusion model from intermedia results
def runFusionIntermediaOneEpoch(trainFusionDataLoader, valFusionDataLoader, fusion_model, optimizer, scheduler_fusion, epoch, global_step, global_val):
  startTime = time.time()
  total_train_loss = 0
  #adjust_learning_rate(optimizer, epoch, args.learning_rate_fusion)
  print("Epoch: {}, Current Stage: Fusion Intermedia, Current Learning Rate: {}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
  # -------------------------------
  # Train ----------------------------------
  for batch_idx, batchData in enumerate(tqdm(trainFusionDataLoader, desc='Train iter {}'.format(epoch))):
    inputs = batchData['inputs'].cuda()
    depthMap = batchData['depthMap'].cuda()
    rgbImgs = batchData['rgbImgs'].cuda() if args.fusion_model == 'withRGB' else None
    soilMask = batchData['soilMask'].cuda() if args.soil_mask_loss else None
    mask = (depthMap > 0) & (~torch.isnan(depthMap)) & (~torch.isinf(depthMap)) & (depthMap <= args.max_depth)
    loss = trainFusionIntermedia(inputs, depthMap, rgbImgs, mask, fusion_model, optimizer, soilMask)
    total_train_loss += loss
    global_step += 1
    writer.add_scalar('loss fusion', loss, global_step)  # tensorboardX for iter
  writer.add_scalar('total fusion train loss', total_train_loss / len(trainFusionDataLoader), epoch)  # tensorboardX for epoch
  print("epoch: {}, avg train loss: {}".format(epoch, total_train_loss / len(trainFusionDataLoader)))
  # ----------------------------------------------------

  # Save Checkpoint ------------------------------------
  saveCkpt(epoch, total_train_loss / len(trainFusionDataLoader), fusion_model, stage='fusion', model_name=args.fusion_model)
  # --------------------------------------------------------

  # Valid --------------------------------------------------
  eType = 'depth'
  error_names = getMetrics(eType)
  numTestData = args.batch_size_fusion * len(valFusionDataLoader)
  errorsPred = np.zeros((len(error_names), numTestData), np.float32)
  weights = torch.ones(1, 1, heightC, widthC).to(device)
  total_val_loss = 0
  total_val_crop_rmse = 0
  counter = 0
  for batch_idx, batchData in enumerate(tqdm(valFusionDataLoader, desc='Train iter {}'.format(epoch))):
    inputs = batchData['inputs'].cuda()
    depthGT = batchData['depthMap'].cuda()
    rgbImgs = batchData['rgbImgs'].cuda() if args.fusion_model == 'withRGB' else None
    mask = (depthGT > 0) & (~torch.isnan(depthGT)) & (~torch.isinf(depthGT)) & (depthGT <= args.max_depth)
    b, c, h, w = inputs.shape
    val_loss, val_output = valFusionIntermedia(inputs, depthGT, rgbImgs, mask, fusion_model)
    if batch_idx == 0:
      saveValOutputSample(val_output, mask, depthGT, epoch)
    errors = compute_errors(depthGT.clone(), val_output, ~mask, weights=weights, type=eType)
    for k in range(len(errors)):
      errorsPred[k, counter:counter + b] = errors[k].squeeze_(-1).squeeze_(-1).squeeze_(-1).cpu()
    counter += b
    val_crop_rmse = torch.sqrt(torch.mean((depthGT[mask] - val_output[mask])**2))
    # -------------------------------------------------------------
    # Loss ---------------------------------
    total_val_loss += val_loss
    total_val_crop_rmse += val_crop_rmse
    # ---------------------------------------
    # Step ------
    global_val += 1
    # ------------
  if scheduler_fusion is not None:
    scheduler_fusion.step()
  writer.add_scalar('total fusion validation loss', total_val_loss / (len(valFusionDataLoader)), epoch)  # tensorboardX for validation in epoch
  writer.add_scalar('total fusion validation rmse', total_val_crop_rmse / (len(valFusionDataLoader)), epoch)  # tensorboardX rmse for validation in epoch
  print("epoch: {}, avg val loss: {}, avg val rmse {}".format(epoch, total_val_loss / len(valFusionDataLoader), total_val_crop_rmse / (len(valFusionDataLoader))))
  mean_errorsPred = errorsPred.mean(1)
  print("validation results (test on epoch: {}, model <{}>): ".format(epoch, args.fusion_model))
  print("\t|" + ("{:^10}|" * len(error_names)).format(*error_names))
  print("\t" + "-" * (len(error_names) * 11))
  print("\t|" + ("{:^10.4f}|" * len(mean_errorsPred)).format(*mean_errorsPred))
  print("Time of This epoch: {} seconds".format(time.time() - startTime))
  return global_step, global_val


# Disparity to Depth Function
def todepth(disp):
  H = 512  # image height
  W = 1024  # image width
  b = 0.2  # baseline
  theta_T = math.pi - ((np.arange(H).astype(np.float64) + 0.5) * math.pi / H)
  theta_T = np.tile(theta_T[:, None], (1, W))
  angle = b * np.sin(theta_T)
  angle2 = b * np.cos(theta_T)
  #################
  for i in range(len(disp)):
    mask = disp[i, :, :] == 0
    de = np.zeros(disp.shape)
    de[i, :, :] = angle / np.tan(disp[i, :, :] / 180 * math.pi) + angle2
    de[i, :, :][mask] = 0
  return de


"""
Main Processing Start From Here
"""
print("basic settings")
# tensorboard Setting -----------------------
curDateTime = datetime.now().strftime("%Y%m%d_%H%M%S")
writerPath = os.path.join(args.tensorboard_path, curDateTime)
imagePath = os.path.join(args.save_image_path, curDateTime)
os.makedirs(writerPath)
os.makedirs(imagePath)
writer = SummaryWriter(writerPath)
# -----------------------------------------
# basic parameter setting
heightE, widthE = args.heightE, args.widthE
heightC, widthC = args.heightC, args.widthC
minTheta = -np.pi / 2
baseline = args.baseline
maxDepth = args.max_depth
# -------------------------------------------------
# Cassini View Trans, Disp to Depth， grid sample
dispDepthT = CassiniViewDepthTransformer()
# -------------------------------------------------
# import dataloader ------------------------------
print("Preparing data. Dataset: <{}>".format(args.dataset))
if args.dataset == 'deep360':
  myDataset = Multi360DepthSoiledDataset if args.soiled else Multi360DepthDataset
  catEquiInfo = True if args.disp_model == '360SDNet' else False
  # disp
  if args.mode == 'all' or args.mode == 'disp':
    trainDispData = myDataset(dataUsage='disparity', crop=args.crop_disp, rootDir=args.dataset_root, curStage='training', catEquiInfo=catEquiInfo)
    valDispData = myDataset(dataUsage='disparity', crop=False, rootDir=args.dataset_root, curStage='validation', catEquiInfo=catEquiInfo)
    trainDispDataLoader = torch.utils.data.DataLoader(trainDispData, batch_size=args.batch_size_disp, num_workers=4, pin_memory=False, shuffle=True)
    valDispDataLoader = torch.utils.data.DataLoader(valDispData, batch_size=args.batch_size_disp, num_workers=4, pin_memory=False, shuffle=False)

  # fusion
  elif args.mode == "intermedia_fusion" or args.mode == "fusion":
    if args.mode == "intermedia_fusion":
      myDataset = Multi360FusionSoiledDataset if args.soiled else Multi360FusionDataset
      trainFusionData = myDataset(rootInputDir=args.intermedia_path, rootDepthDir=args.dataset_root, curStage='training', shuffleOrder=args.fusion_shuffle_order, needMask=args.soil_mask_loss)
      valFusionData = myDataset(rootInputDir=args.intermedia_path, rootDepthDir=args.dataset_root, curStage='validation', shuffleOrder=args.fusion_shuffle_order)
    elif args.mode == "fusion":
      trainFusionData = myDataset(dataUsage='fusion', crop=False, curStage='training', catEquiInfo=catEquiInfo)
      valFusionData = myDataset(dataUsage='fusion', crop=False, curStage='validation', catEquiInfo=catEquiInfo)
    trainFusionDataLoader = torch.utils.data.DataLoader(trainFusionData, batch_size=args.batch_size_fusion, num_workers=4, pin_memory=False, shuffle=True)
    valFusionDataLoader = torch.utils.data.DataLoader(valFusionData, batch_size=args.batch_size_fusion, num_workers=4, pin_memory=False, shuffle=False)
  else:
    raise NotImplementedError("mode <{}> is not supported!".format(args.mode))
elif args.dataset == "OmniHouse" or args.dataset == "Sunny" or args.dataset == "Cloudy" or args.dataset == "Sunset":
  myDataset = OmniFisheyeDataset
  # disp
  if args.mode == 'all' or args.mode == 'disp':
    trainDispData = myDataset(mode='disparity',
                              curStage='training',
                              datasetName=args.dataset,
                              shape=(640,
                                     320),
                              crop=False,
                              catEquiInfo=False,
                              soiled=False,
                              shuffleOrder=False,
                              inputRGB=True,
                              needMask=False)
    valDispData = myDataset(mode='disparity',
                              curStage='testing',# fish eye datasets have no validation set, using testing set
                              datasetName=args.dataset,
                              shape=(640,
                                      320),
                              crop=False,
                              catEquiInfo=False,
                              soiled=False,
                              shuffleOrder=False,
                              inputRGB=True,
                              needMask=False)
    trainDispDataLoader = torch.utils.data.DataLoader(trainDispData, batch_size=args.batch_size_disp, num_workers=4, pin_memory=False, shuffle=True)
    valDispDataLoader = torch.utils.data.DataLoader(valDispData, batch_size=args.batch_size_disp, num_workers=4, pin_memory=False, shuffle=False)
  elif args.mode == "intermedia_fusion" or args.mode == "fusion":
    trainFusionData = myDataset(mode=args.mode,
                                curStage='training',
                                datasetName=args.dataset,
                                shape=(640,
                                       320),
                                crop=False,
                                catEquiInfo=False,
                                soiled=False,
                                shuffleOrder=False,
                                inputRGB=True,
                                needMask=False)
    valFusionData = myDataset(mode=args.mode,
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
    trainFusionDataLoader = torch.utils.data.DataLoader(trainFusionData, batch_size=args.batch_size_fusion, num_workers=4, pin_memory=False, shuffle=True)
    valFusionDataLoader = torch.utils.data.DataLoader(valFusionData, batch_size=args.batch_size_fusion, num_workers=4, pin_memory=False, shuffle=False)
  else:
    raise NotImplementedError("mode <{}> is not supported!".format(args.mode))
elif args.dataset == '3d60':
  myDataset = Dataset3D60
  if args.mode == 'all' or args.mode == 'disp':
    trainDispData = myDataset(filenamesFile='./dataloader/3d60_train.txt',
                              rootDir=args.dataset_root,
                              mode='disparity',
                              curStage='training',
                              shape=(512,
                                     256),
                              crop=False,
                              catEquiInfo=False,
                              soiled=False,
                              shuffleOrder=False,
                              inputRGB=True,
                              needMask=False)
    valDispData = myDataset(filenamesFile='./dataloader/3d60_val.txt',
                              rootDir=args.dataset_root,
                              mode='disparity',
                              curStage='validation',#
                              shape=(512,
                                      256),
                              crop=False,
                              catEquiInfo=False,
                              soiled=False,
                              shuffleOrder=False,
                              inputRGB=True,
                              needMask=False)
    trainDispDataLoader = torch.utils.data.DataLoader(trainDispData, batch_size=args.batch_size_disp, num_workers=4, pin_memory=False, shuffle=True)
    valDispDataLoader = torch.utils.data.DataLoader(valDispData, batch_size=args.batch_size_disp, num_workers=4, pin_memory=False, shuffle=False)
  elif args.mode == "intermedia_fusion" or args.mode == "fusion":
    trainFusionData = myDataset(filenamesFile='./dataloader/3d60_train.txt',
                                rootDir=args.dataset_root,
                                interDir=args.intermedia_path,
                                mode=args.mode,
                                curStage='training',
                                datasetName=args.dataset,
                                shape=(640,
                                       320),
                                crop=False,
                                catEquiInfo=False,
                                soiled=False,
                                shuffleOrder=False,
                                inputRGB=True,
                                needMask=False)
    valFusionData = myDataset(filenamesFile='./dataloader/3d60_val.txt',
                              rootDir=args.dataset_root,
                              interDir=args.intermedia_path,
                              mode=args.mode,
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
    trainFusionDataLoader = torch.utils.data.DataLoader(trainFusionData, batch_size=args.batch_size_fusion, num_workers=4, pin_memory=False, shuffle=True)
    valFusionDataLoader = torch.utils.data.DataLoader(valFusionData, batch_size=args.batch_size_fusion, num_workers=4, pin_memory=False, shuffle=False)
else:
  raise NotImplementedError("Dataset <{}> is not supported yet!".format(args.dataset))
# -------------------------------------------------

# Define models ----------------------------------------------

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
  if args.disp_model in dispModelDict:
    disp_model = dispModelDict[args.disp_model](args.max_disp)
  else:
    raise NotImplementedError('Required Model {} is Not Implemented!!!'.format(args.disp_model))
else:
  disp_model = None
if args.mode == 'intermedia_fusion' or args.mode == 'fusion':
  if args.fusion_model in fusionModelDict:
    fusion_model = fusionModelDict[args.fusion_model](max_depth=args.max_depth)
  else:
    raise NotImplementedError('Required Model {} is Not Implemented!!!'.format(args.fusion_model))
else:
  fusion_model = None
# ----------------------------------------------------------

# assign initial value of filter cost volume for 360SDNet ---------------------------------
if args.disp_model == '360SDNet':
  init_array = np.zeros((1, 1, 7, 1))  # 7 of filter
  init_array[:, :, 3, :] = 28. / 540
  init_array[:, :, 2, :] = 512. / 540
  disp_model.forF.forfilter1.weight = torch.nn.Parameter(torch.Tensor(init_array))

  # if use nn.DataParallel(model), model.module.filtercost
# else use model.filtercost
if args.disp_model == '360SDNet':
  if args.parallel:
    freeze_layer(disp_model.module.forF.forfilter1)
  else:
    freeze_layer(disp_model.forF.forfilter1)
# -------------------------------------------------

# Load Checkpoint -------------------------------
start_epoch = 0

# Checkpoint ----------
if not os.path.isdir(args.save_checkpoint_path):
  os.makedirs(args.save_checkpoint_path)

# Load ckpt or init model
if (disp_model is not None):
  initType = 'default'
  initModel(disp_model, initType)
  if (args.checkpoint_disp is not None) and (args.checkpoint_disp != 'None'):
    if not args.loadSHG:
      checkpoint_disp = torch.load(args.checkpoint_disp)
      if 'state_dict' in checkpoint_disp.items():
        disp_model.load_state_dict(checkpoint_disp['state_dict'])
      else:
        disp_model.load_state_dict(checkpoint_disp)
      print("load disparity model <{}> from <{}>".format(args.disp_model, args.checkpoint_disp))
    else:
      # for n, p in disp_model.named_parameters():
      #   if n == 'module.feature_extraction.layer3.2.conv1.0.0.weight': print(n, ':', p)
      if args.disp_model == '360SDNet':
        loadPartialModel360SD(disp_model, args.checkpoint_disp)
      else:
        loadPartialModel(disp_model, args.checkpoint_disp)
      # for n, p in disp_model.named_parameters():
      #   if n == 'module.feature_extraction.layer3.2.conv1.0.0.weight': print(n, ':', p)
      print("load part of disparity model <{}> from <{}>".format(args.disp_model, args.checkpoint_disp))
      if args.freezeSHG:
        freezeSHG(disp_model)
        print("freeze stack hourglass part")
  else:
    print("initialize model <{}> as type <{}>".format(args.disp_model, initType))
if (fusion_model is not None):
  if (args.checkpoint_fusion is not None) and (args.checkpoint_fusion != 'None'):
    checkpoint_fusion = torch.load(args.checkpoint_fusion)
    fusion_model.load_state_dict(checkpoint_fusion['state_dict'])
    print("load fusion model <{}> from <{}>".format(args.fusion_model, args.checkpoint_fusion))
  else:
    initType = 'kaiming_uniform'
    initModel(fusion_model, initType)
    print("initialize model <{}> as type <{}>".format(args.fusion_model, initType))
if args.parallel:
  if disp_model is not None: disp_model = nn.DataParallel(disp_model)
  if fusion_model is not None: fusion_model = nn.DataParallel(fusion_model)
if args.cuda:
  if disp_model is not None: disp_model.cuda()
  if fusion_model is not None: fusion_model.cuda()
# Optimizer ----------
if args.mode == 'all' or args.mode == 'disp':
  optimizer_disp = optim.Adam(disp_model.parameters(), lr=args.learning_rate_disp, betas=(0.9, 0.999))
  if args.scheduler:
    scheduler_disp = optim.lr_scheduler.StepLR(optimizer=optimizer_disp, step_size=20, gamma=0.5)
    print("disp scheduler set!")
  else:
    scheduler_disp = None
if args.mode == 'all' or args.mode == 'fusion' or args.mode == 'intermedia_fusion':
  optimizer_fusion = optim.Adam(fusion_model.parameters(), lr=args.learning_rate_fusion, betas=(0.9, 0.999))
  #optimizer_fusion = optim.SGD(fusion_model.parameters(), lr=0.0001)
  if args.scheduler:
    scheduler_fusion = optim.lr_scheduler.StepLR(optimizer=optimizer_fusion, step_size=20, gamma=0.8)
  else:
    scheduler_fusion = None

# ---------------------


# Main Function ----------------------------------
def main():
  print("Training Start!!!")
  global_step = 0
  global_val = 0
  # Multi_GPU for model ----------------------------

  # Start Training -----------------------------
  start_full_time = time.time()
  # for epoch in tqdm(range(start_epoch + 1, args.epochs + 1), desc='Epoch'):
  for epoch in range(start_epoch + 1, args.epochs + 1):
    if args.mode == 'all':
      if epoch < (args.epochs // 5 * 3) + 1:
        global_step, global_val = runDispOneEpoch(trainDispDataLoader, valDispDataLoader, disp_model, optimizer_disp, scheduler_disp, epoch, global_step, global_val)
      else:
        global_step, global_val = runFusionOneEpoch(trainFusionDataLoader, valFusionDataLoader, disp_model, fusion_model, optimizer_fusion, epoch, global_step, global_val)
    elif args.mode == 'disp':
      if args.disp_model == '360SDNet' and epoch >= args.start_learn:
        unfreeze_layer(disp_model.module.forF.forfilter1)
      if args.freezeSHG and epoch >= args.start_learn:
        unfreezeSHG(disp_model)
        print("unfreeze stack hourglass")
      global_step, global_val = runDispOneEpoch(trainDispDataLoader, valDispDataLoader, disp_model, optimizer_disp, scheduler_disp, epoch, global_step, global_val)
    elif args.mode == 'fusion':
      global_step, global_val = runFusionOneEpoch(trainFusionDataLoader, valFusionDataLoader, disp_model, fusion_model, optimizer_fusion, epoch, global_step, global_val)
    elif args.mode == 'intermedia_fusion':
      global_step, global_val = runFusionIntermediaOneEpoch(trainFusionDataLoader, valFusionDataLoader, fusion_model, optimizer_fusion, scheduler_fusion, epoch, global_step, global_val)
  writer.close()
  # End Training
  print("Training Ended!!!")
  print('full training time = %.2f Hours' % ((time.time() - start_full_time) / 3600))


# ----------------------------------------------------------------------------

if __name__ == '__main__':
  main()
