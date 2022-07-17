from __future__ import print_function
import argparse
from ast import arg
import os
import os.path as osp
import re
import random
from utils.geometry import cassini2Equirec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
import cv2
from dataloader import list_file as lt
from dataloader import deep360_loader as DA
from models import Baseline, ModeFusion
from utils import evaluation
import prettytable as pt

parser = argparse.ArgumentParser(description='MODE Fusion testing')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='maximum depth in meters')
parser.add_argument('--model', default='ModeFusion', help='select model')
parser.add_argument('--dbname', default="Deep360", help='dataset name')
parser.add_argument('--soiled', action='store_true', default=False, help='test fusion network on soiled data (only for Deep360)')
parser.add_argument('--resize', action='store_true', default=False, help='resize the input by downsampling to 1/2 of its original size')
parser.add_argument('--datapath-input', default='./outputs/Deep360PredDepth/', help='the path of the input of stage2, which is just the output of stage1')
parser.add_argument('--datapath-dataset', default='./datasets/Deep360/', help='the path of the dataset')
parser.add_argument('--outpath', default='./MODE_Fusion_output/', help='the output path for fusion results')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--loadmodel', default=None, help='load model path')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)

if args.dbname == 'Deep360':
  test_depthes, test_confs, test_rgbs, test_gt = lt.list_deep360_fusion_test(args.datapath_input, args.datapath_dataset, args.soiled)

if args.model == 'Baseline':
  model = Baseline(args.maxdepth)
elif args.model == 'ModeFusion':
  if args.dbname == 'Deep360':
    model = ModeFusion(args.maxdepth, [32, 64, 128, 256], {'depth': 12, 'rgb': 12})
else:
  print('no model')

if args.cuda:
  model = nn.DataParallel(model)
  model.cuda()

if args.loadmodel is not None:
  print('Load pretrained model')
  pretrain_dict = torch.load(args.loadmodel)
  model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def test(depthes, confs, rgbs, gt):
  model.eval()

  if args.cuda:
    depthes = [depth.cuda() for depth in depthes]
    confs = [conf.cuda() for conf in confs]
    rgbs = [rgb.cuda() for rgb in rgbs]
    gt = gt.cuda()

  with torch.no_grad():
    if args.model == 'Baseline':
      output = model(depthes)
    elif args.model == 'ModeFusion':
      output = model(depthes, confs, rgbs)
    if args.resize:
      output = F.interpolate(output, scale_factor=[2, 2], mode='bicubic', align_corners=True)
    pred = torch.squeeze(output, 1)

  # Convert the pred depth map and gt in Cassini domain to ERP domain
  pred = cassini2Equirec(pred.unsqueeze(1))
  gt = cassini2Equirec(gt.unsqueeze(1))

  #---------
  mask = gt <= args.maxdepth
  #----
  eval_metrics = []
  eval_metrics.append(evaluation.mae(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.rmse(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.absrel(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.sqrel(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.silog(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.delta_acc(1, pred[mask], gt[mask]))
  eval_metrics.append(evaluation.delta_acc(2, pred[mask], gt[mask]))
  eval_metrics.append(evaluation.delta_acc(3, pred[mask], gt[mask]))

  return np.array(eval_metrics), pred.data.cpu().numpy(), gt.data.cpu().numpy()


def main():
  TestImgLoader = torch.utils.data.DataLoader(DA.Deep360DatasetFusion(test_depthes,
                                                                      test_confs,
                                                                      test_rgbs,
                                                                      test_gt,
                                                                      args.resize,
                                                                      False),
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.batch_size,
                                              drop_last=False)
  #------------- Check the output directories -------
  snapshot_name = osp.splitext(osp.basename(args.loadmodel))[0]
  result_dir = osp.join(args.outpath, args.dbname, snapshot_name)
  depth_pred_path = os.path.join(result_dir, "depth_pred")
  if not os.path.exists(depth_pred_path):
    os.makedirs(depth_pred_path)
  gt_png_path = os.path.join(result_dir, "gt_png")
  if not os.path.exists(gt_png_path):
    os.makedirs(gt_png_path)
  #------------- TESTING -------------------------
  total_eval_metrics = np.zeros(8)
  for batch_idx, (gt_name, depthes, confs, rgbs, gt) in enumerate(TestImgLoader):
    print("\rStage2 Test: {:.2f}%".format(100 * (batch_idx + 1) / len(TestImgLoader)), end='')

    eval_metrics, depth_pred_batch, gt_batch = test(depthes, confs, rgbs, gt)
    total_eval_metrics += eval_metrics

    for i in range(depth_pred_batch.shape[0]):
      name = osp.splitext(osp.basename(gt_name[i]))[0]
      if args.dbname == 'Deep360':
        ep_name = re.findall(r'ep[0-9]_', gt_name[i])[0]
        name = ep_name + name

      # save gt png
      depth_gt = gt_batch[i]
      depth_gt = np.log(depth_gt - np.min(depth_gt) + 1)
      depth_gt = 255 * depth_gt / np.max(depth_gt)
      depth_gt = np.clip(depth_gt, 0, 255)
      depth_gt = depth_gt.astype(np.uint8)
      depth_gt = cv2.applyColorMap(depth_gt, cv2.COLORMAP_JET)
      cv2.imwrite(gt_png_path + '/' + name + "_gt.png", depth_gt)

      # save depth pred
      depth_pred = depth_pred_batch[i]
      np.save(depth_pred_path + '/' + name + "_pred.npy", depth_pred)
      depth_pred = np.log(depth_pred - np.min(depth_pred) + 1)
      depth_pred = 255 * depth_pred / np.max(depth_pred)
      depth_pred = np.clip(depth_pred, 0, 255)
      depth_pred = depth_pred.astype(np.uint8)
      depth_pred = cv2.applyColorMap(depth_pred, cv2.COLORMAP_JET)
      cv2.imwrite(depth_pred_path + '/' + name + "_pred.png", depth_pred)

  eval_metrics = total_eval_metrics / len(TestImgLoader)
  tb = pt.PrettyTable()
  tb.field_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
  tb.add_row(list(eval_metrics))
  print('\nTest Results:\n')
  print(tb)


if __name__ == '__main__':
  main()
