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

from models import ModeDisparity
from models import initModelPara, loadStackHourglassOnly

from utils import evaluation
from dataloader import listfile_disparity_train, Deep360DatasetDisparity
'''
Argument Definition
'''

parser = argparse.ArgumentParser(description='MODE Disparity estimation - training')

# model
parser.add_argument('--model_disp', default='ModeDisparity', help='select model')
# data
parser.add_argument("--dataset", default="deep360", type=str, help="dataset name")
parser.add_argument("--dataset_root", default="../../datasets/MODE_Datasets/Deep360/", type=str, help="dataset root directory.")
parser.add_argument('--width', default=512, type=int, help="width of omnidirectional images in Cassini domain")
parser.add_argument('--height', default=1024, type=int, help="height of omnidirectional images in Cassini domain")
# stereo
parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--max_depth', default=1000, type=float, help="max valid depth")
parser.add_argument('--baseline', default=1, type=float, help="baseline of binocular spherical system")
# hyper parameters
parser.add_argument('--epochs', type=int, default=55, help='number of epochs to train')
parser.add_argument('--start_decay', type=int, default=45, help='number of epoch for lr to start decay')
parser.add_argument('--batch_size', type=int, default=4, help='number of batch to train')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate of disp estimation training')

# training
parser.add_argument('--resume', action='store_true', default=False, help='load checkpoint and resume learning')
parser.add_argument('--checkpoint_disp', default=None, help='load checkpoint of disparity estimation path')
parser.add_argument('--loadSHGonly', action='store_true', default=False, help='if set,load stack hour glass part from pretrained model, skip feature extraction part')
parser.add_argument('--pretrained', default=None, help='load pretrained disp model path')
parser.add_argument('--tensorboard_path', default='./logs', help='tensorboard path')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--printNet', action='store_true', default=False, help='print network details')
parser.add_argument('--crop_disp', action='store_true', default=False, help='crop when train disp')
parser.add_argument('--parallel', action='store_true', default=False, help='train model parallel')
parser.add_argument('--cudnn_deter', action='store_true', default=False, help='if True, set cudnn deterministic as True and benchmark as False. Otherwise the opposite')
parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 123)')

parser.add_argument('--soiled', action='store_true', default=False, help='test soiled image')

# saving
parser.add_argument('--save_suffix_disp', type=str, default=None, help='save checkpoint name')
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
def trainDisp(imgL, imgR, disp_true, mask, model_disp, disp_optimizer):
  model_disp.train()
  disp_optimizer.zero_grad()
  # Loss --------------------------------------------
  output1, output2, output3 = model_disp(imgL, imgR)
  loss = 0.5 * F.smooth_l1_loss(output1[mask],
                                disp_true[mask],
                                size_average=True) + 0.7 * F.smooth_l1_loss(output2[mask],
                                                                            disp_true[mask],
                                                                            size_average=True) + F.smooth_l1_loss(output3[mask],
                                                                                                                  disp_true[mask],
                                                                                                                  size_average=True)
  # --------------------------------------------------
  loss.backward()
  disp_optimizer.step()

  return loss.data.item()


def valDisp(imgL, imgR, disp_true, mask, model_disp):
  model_disp.eval()

  with torch.no_grad():
    output = model_disp(imgL, imgR)
    if len(disp_true[mask]) == 0:
      loss = 0
    else:
      loss = torch.mean(torch.abs(output[mask] - disp_true[mask]))  # end-point-error
    d1 = evaluation.D1(th_pixel=3, th_pct=0.05, pred=output[mask], gt=disp_true[mask])

  return loss, d1, output


def train(trainDispDataLoader, valDispDataLoader, model_disp, optimizer):
  print("Training Start!!!")
  global_step = 0
  global_val = 0
  for epoch in range(start_epoch + 1, args.epochs + 1):
    startTime = time.time()
    total_train_loss = 0
    counter = 0
    adjust_learning_rate(optimizer, epoch, args.learning_rate_disp)
    print("Epoch: {}, Current Stage: Disp, Current Learning Rate: {}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    # -------------------------------
    # Train ----------------------------------
    for batch_idx, batchData in enumerate(tqdm(trainDispDataLoader, desc='Train iter {}'.format(epoch))):
      leftImg = batchData['leftImg'].cuda()
      rightImg = batchData['rightImg'].cuda()
      dispMap = batchData['dispMap'].cuda()
      mask = (~torch.isnan(dispMap))
      # for fish eye datasets
      b, c, h, w = leftImg.shape
      loss = trainDisp(leftImg, rightImg, dispMap, mask, model_disp, optimizer)
      counter += b
      total_train_loss += loss
      global_step += 1
      writer.add_scalar('loss disp', loss, global_step)  # tensorboardX for iter
    writer.add_scalar('total disp train loss', total_train_loss / len(trainDispDataLoader), epoch)  # tensorboardX for epoch
    print("epoch: {}, avg train loss: {}".format(epoch, total_train_loss / len(trainDispDataLoader)))
    # ----------------------------------------------------

    # Save Checkpoint ------------------------------------
    saveCkpt(epoch, total_train_loss / len(trainDispDataLoader), model_disp, stage='disp', model_name=args.model_disp)
    # --------------------------------------------------------

    # Valid --------------------------------------------------
    total_val_loss = 0
    total_val_d1 = 0
    counter = 0
    for batch_idx, batchData in enumerate(tqdm(valDispDataLoader, desc='Train iter {}'.format(epoch))):
      leftImg = batchData['leftImg'].cuda()
      rightImg = batchData['rightImg'].cuda()
      dispMap = batchData['dispMap'].cuda()
      mask = (dispMap > 0) & (~torch.isnan(dispMap)) & (~torch.isinf(dispMap)) & (dispMap <= args.max_disp)
      b, c, h, w = leftImg.shape
      val_loss, val_d1, val_output = valDisp(leftImg, rightImg, dispMap, mask, model_disp)
      if batch_idx == 0:  # save validation sample
        saveValOutputSample(val_output, mask, dispMap, epoch)
      counter += b
      # Loss ---------------------------------
      total_val_loss += val_loss
      total_val_d1 += val_d1
      # Step ------
      global_val += 1
      # ------------
    writer.add_scalar('total disp validation loss', total_val_loss / counter, epoch)  # tensorboardX for validation in epoch
    writer.add_scalar('total disp validation d1', total_val_d1 / counter, epoch)  # tensorboardX rmse for validation in epoch
    print("epoch: {}, avg val loss: {}, avg val d1 {}".format(epoch, total_val_loss / counter, total_val_d1 / counter))
    print("Time of This epoch: {} seconds".format(time.time() - startTime))


"""
Main Processing Start From Here
"""
print("basic settings")
# tensorboard Setting -----------------------
curDateTime = datetime.now().strftime("%Y%m%d_%H%M%S")
writerPath = os.path.join(args.tensorboard_path, curDateTime)
imagePath = os.path.join(args.save_image_path, curDateTime)
os.makedirs(writerPath, exist_ok=True)  # log
os.makedirs(imagePath, exist_ok=True)  # image sample
os.makedirs(args.save_checkpoint_path, exist_ok=True)  # checkpoint
writer = SummaryWriter(writerPath)
# -------------------------------------------------
# import dataloader ------------------------------
print("Preparing data. Dataset: <{}>".format(args.dataset))
if args.dataset == 'deep360':
  train_left_img, train_right_img, train_left_disp, val_left_img, val_right_img, val_left_disp = listfile_disparity_train(args.dataset_root, soiled=args.soiled)
  trainDispData = Deep360DatasetDisparity()
  valDispData = Deep360DatasetDisparity()
  trainDispDataLoader = torch.utils.data.DataLoader(trainDispData, batch_size=args.batch_size, num_workers=4, pin_memory=False, shuffle=True)
  valDispDataLoader = torch.utils.data.DataLoader(valDispData, batch_size=args.batch_size, num_workers=4, pin_memory=False, shuffle=False)
# -------------------------------------------------

# Define models ----------------------------------------------
model_disp = ModeDisparity(maxdisp=args.max_disp, conv='Sphere', in_height=args.height, in_width=args.width, sphereType='Cassini', out_conf=False)
# ----------------------------------------------------------
if args.parallel:
  model_disp = nn.DataParallel(model_disp)
if args.cuda:
  model_disp.cuda()

# Load Checkpoint -------------------------------
start_epoch = 0
# Load ckpt or init model
if (model_disp is not None):
  initType = 'default'
  initModel(model_disp, initType)
  if (args.checkpoint_disp is not None) and (args.checkpoint_disp != 'None'):
    if args.resume:
      model_disp, start_epoch = loadCkpt(model_disp, args.checkpoint_disp)
    else:
      if not args.loadSHGonly:  # load all parameters
        checkpoint_disp = torch.load(args.checkpoint_disp)
        if 'state_dict' in checkpoint_disp.items():
          model_disp.load_state_dict(checkpoint_disp['state_dict'])
        else:
          model_disp.load_state_dict(checkpoint_disp)
        print("load disparity model <{}> from <{}>".format(args.model_disp, args.checkpoint_disp))
      else:  # laod stack hourglass only
        loadStackHourglassOnly(model_disp, args.checkpoint_disp)
        print("load stackhourglass part of disparity model <{}> from <{}>".format(args.model_disp, args.checkpoint_disp))
  else:
    print("initialize model <{}> as type <{}>".format(args.model_disp, initType))

# Optimizer ----------
optimizer_disp = optim.Adam(model_disp.parameters(), lr=args.learning_rate_disp, betas=(0.9, 0.999))

# ---------------------


# Main Function ----------------------------------
def main():
  print("Training Start!!!")
  # Start Training -----------------------------
  start_full_time = time.time()
  train(trainDispDataLoader, valDispDataLoader, model_disp, optimizer_disp)
  writer.close()
  # End Training
  print("Training Ended!!!")
  print('full training time = %.2f Hours' % ((time.time() - start_full_time) / 3600))


# ----------------------------------------------------------------------------

if __name__ == '__main__':
  main()