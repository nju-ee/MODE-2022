import os

import argparse
import torch
import numpy as np

defaultMetrics = ['D1 %', 'px5 %', 'px3 %', 'px1 %', 'rmse', 'mae', 'log_rmse', 'SILog', 'abs_rel', 'sq_rel', 'delta1 %', 'delta2 %', 'delta3 %']
dispMetrics = defaultMetrics[0:6]
depthMetrics = defaultMetrics[4:]

evaTypeDict = {'all': defaultMetrics, 'disp': dispMetrics, 'depth': depthMetrics}


def getMetrics(type):
  return evaTypeDict[type]


class evaluator():
  def __init__(self, numData, type):
    assert type in evaTypeDict

  def __calDelta(self, pred, gt, invalid_mask, valid_sum, sum_dims):
    thresh = torch.max((gt / pred), (pred / gt))
    thresh[invalid_mask] = 2.0
    delta1 = (thresh < 1.25).float().sum(dim=sum_dims, keepdim=True).float() / valid_sum.float()
    delta2 = (thresh < (1.25**2)).float().sum(dim=sum_dims, keepdim=True).float() / valid_sum.float()
    delta3 = (thresh < (1.25**3)).float().sum(dim=sum_dims, keepdim=True).float() / valid_sum.float()
    return delta1, delta2, delta3

  def __calRmse(self, pred, gt, invalid_mask, valid_sum, sum_dims, weights):
    rmse = (gt - pred)**2
    rmse[invalid_mask] = 0.0
    rmse_w = rmse * weights
    rmse_mean = torch.sqrt(rmse_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())
    return rmse_mean

  def __calRmseLog(self, pred, gt, invalid_mask, valid_sum, sum_dims, weights):
    rmse_log = (torch.log(gt) - torch.log(pred))**2
    rmse_log[invalid_mask] = 0.0
    rmse_log_w = rmse_log * weights
    rmse_log_mean = torch.sqrt(rmse_log_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())
    return rmse_log_mean

  def __calSilog(self, pred, gt, invalid_mask, valid_sum, sum_dims, weights):
    # scaleInvariantError
    sie = torch.log(pred) - torch.log(gt)
    sie[invalid_mask] = 0.0
    sie_mean = torch.mean(sie**2, dim=sum_dims, keepdim=True) - ((torch.mean(sie, dim=sum_dims, keepdim=True))**2)
    return sie_mean

  def __calMae(self, pred, gt, invalid_mask, valid_sum, sum_dims, weights):
    mae = torch.abs(gt - pred)
    mae[invalid_mask] = 0.0
    mae_w = mae * weights
    mae_mean = mae_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()
    return mae_mean

  def __calD1(self, pred, gt, invalid_mask, valid_sum, sum_dims, weights):
    pass


def create_image_grid(width, height, data_type=torch.float32):
  v_range = (
      torch.arange(0,
                   height)  # [0 - h]
      .view(1,
            height,
            1)  # [1, [0 - h], 1]
      .expand(1,
              height,
              width)  # [1, [0 - h], W]
      .type(data_type)  # [1, H, W]
  )
  u_range = (
      torch.arange(0,
                   width)  # [0 - w]
      .view(1,
            1,
            width)  # [1, 1, [0 - w]]
      .expand(1,
              height,
              width)  # [1, H, [0 - w]]
      .type(data_type)  # [1, H, W]
  )
  return torch.stack((u_range, v_range), dim=1)  # [1, 2, H, W]


def compute_errors(gt, pred, invalid_mask, weights, type='all'):
  b, _, __, ___ = gt.size()
  valid_sum = torch.sum(~invalid_mask, dim=[1, 2, 3], keepdim=True)
  gt[invalid_mask] = 0.0
  pred[invalid_mask] = 0.0
  thresh = torch.max((gt / pred), (pred / gt))
  thresh[invalid_mask] = 2.0

  sum_dims = [1, 2, 3]
  delta_valid_sum = torch.sum(~invalid_mask, dim=[1, 2, 3], keepdim=True)
  delta1 = (thresh < 1.25).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
  delta2 = (thresh < (1.25**2)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
  delta3 = (thresh < (1.25**3)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
  delta1 = delta1 * 100
  delta2 = delta2 * 100
  delta3 = delta3 * 100

  rmse = (gt - pred)**2
  rmse[invalid_mask] = 0.0
  rmse_w = rmse * weights
  rmse_mean = torch.sqrt(rmse_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

  # deal with LOG carefully
  rmse_log = (torch.log(gt) - torch.log(pred))**2  # to make sure values are greater 1 for log operation
  rmse_log[invalid_mask] = 0.0
  rmse_log_w = rmse_log * weights
  rmse_log_mean = torch.sqrt(rmse_log_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

  # scaleInvariantError
  sie = torch.log(pred) - torch.log(gt)
  sie[invalid_mask] = 0.0
  sieL = []
  for i in range(b):
    sie_one = sie[i, ::]
    mask_one = ~invalid_mask[i, ::]
    sieL.append((torch.sqrt(torch.mean((sie_one[mask_one])**2) - ((torch.mean((sie_one[mask_one])))**2))).repeat(1, 1, 1, 1))
  #sie = torch.sqrt(torch.mean((sie[~invalid_mask])**2) - ((torch.mean((sie[~invalid_mask])))**2))
  sie = torch.cat(sieL, 0)
  #print(sie.shape)

  mae = torch.abs(gt - pred)
  mae[invalid_mask] = 0.0
  mae_w = mae * weights
  mae_mean = mae_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

  d1_error = torch.abs(gt - pred)
  d1_error[invalid_mask] = 0.0
  d1_wrong_mask = (d1_error > 3) & (d1_error > (gt * 0.05))
  d1_error[d1_wrong_mask] = 1.0
  d1_error[~d1_wrong_mask] = 0.0
  d1_error_mean = d1_error.sum(dim=sum_dims, keepdim=True) / valid_sum.float()
  d1_error_mean = d1_error_mean * 100

  px5_error = torch.abs(gt - pred)
  px5_error[invalid_mask] = 0.0
  px5_error[px5_error <= 5] = 0.0
  px5_error[px5_error > 5] = 1.0
  px5_error_mean = px5_error.sum(dim=sum_dims, keepdim=True) / valid_sum.float()
  px5_error_mean = px5_error_mean * 100

  px3_error = torch.abs(gt - pred)
  px3_error[invalid_mask] = 0.0
  px3_error[px3_error <= 3] = 0.0
  px3_error[px3_error > 3] = 1.0
  px3_error_mean = px3_error.sum(dim=sum_dims, keepdim=True) / valid_sum.float()
  px3_error_mean = px3_error_mean * 100

  px1_error = torch.abs(gt - pred)
  px1_error[invalid_mask] = 0.0
  px1_error[px1_error <= 1] = 0.0
  px1_error[px1_error > 1] = 1.0
  px1_error_mean = px1_error.sum(dim=sum_dims, keepdim=True) / valid_sum.float()
  px1_error_mean = px1_error_mean * 100

  abs_rel = (torch.abs(gt - pred) / gt)
  abs_rel[invalid_mask] = 0.0
  abs_rel_w = abs_rel * weights
  abs_rel_mean = abs_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

  sq_rel = (((gt - pred)**2) / (gt**2))  # use KITTI definition of sq_rel
  sq_rel[invalid_mask] = 0.0
  sq_rel_w = sq_rel * weights
  sq_rel_mean = sq_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

  if type == 'all':
    return d1_error_mean, px5_error_mean, px3_error_mean, px1_error_mean, rmse_mean, mae_mean, rmse_log_mean, sie, abs_rel_mean, sq_rel_mean, delta1, delta2, delta3
  elif type == 'disp':
    return d1_error_mean, px5_error_mean, px3_error_mean, px1_error_mean, rmse_mean, mae_mean
  elif type == 'depth':
    return rmse_mean, mae_mean, rmse_log_mean, sie, abs_rel_mean, sq_rel_mean, delta1, delta2, delta3
