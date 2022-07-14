import os
import numpy as np
from numpy.core.fromnumeric import size
import torch
import torch.nn.functional as F


# Losses using in Disparity Networks
def psmnetSHGLoss(input, target, mask, training=True):
  disp_loss = 0
  if training:
    weights = [0.5, 0.7, 1.0]
    assert len(input) == 3
    for k in range(len(input)):
      disp_loss += weights[k] * (F.smooth_l1_loss(input[k][mask], target[mask], size_average=True))
  else:
    disp_loss = F.smooth_l1_loss(input[-1][mask], target[mask], size_average=True)
  return disp_loss


def aanetMultiScaleLoss(input, target, mask, height=1024, width=512, highestOnly=False, sphere=False):
  # input = list of disp at scales of H/12, H/6, H/3, H/2, H
  disp_loss = 0
  numPred = 4 if sphere else 5
  if not highestOnly:
    assert len(input) == numPred
    if sphere:
      pyramid_weight = [1 / 4, 1 / 2, 1.0, 2.0]  # AANet and AANet+
      scales = [1 / 8, 1 / 4, 1 / 2, 1]
    else:
      pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]  # AANet and AANet+
      scales = [1 / 12, 1 / 6, 1 / 3, 1 / 2, 1]
    for k in range(len(input)):
      pred = input[k]
      # TODO
      # need to fixed
      pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=False) / scales[k]
      disp_loss += pyramid_weight[k] * (F.smooth_l1_loss(pred[mask], target[mask], reduction='mean'))
  else:
    pred = input[-1]
    disp_loss = F.smooth_l1_loss(pred[mask], target[mask], reduction='mean')
  return disp_loss


# Losses using in Fusion Networks
def scaleInvariantLoss(input, target, mask, lam):
  D = torch.log(input[mask]) - torch.log(target[mask])
  loss = torch.mean(D**2) - lam * ((torch.mean(D))**2)
  return loss


def scaleInvariantLossWithSoilMask(input, target, mask, soilMask, lam, rate=3):
  assert soilMask is not None
  D = torch.log(input[mask]) - torch.log(target[mask])
  weight = soilMask * (rate - 1) + 1.0  # soiled/other = 3/1
  D = weight[mask] * D
  loss = torch.mean(D**2) - lam * ((torch.mean(D))**2)
  return loss


def MultiscaleInvariantLoss(inputs, target, mask, lam):
  loss = 0
  scales = [1 / 4, 1 / 2, 1]
  weights = [0.5, 0.7, 1.0]
  assert (len(inputs) == len(scales))
  for i in range(len(inputs)):
    input = F.interpolate(inputs[i], scale_factor=1 / (scales[i]))
    D = torch.log(input[mask]) - torch.log(target[mask])
    loss += (torch.mean(D**2) - lam * ((torch.mean(D))**2)) * weights[i]
  return loss


def BerHuLoss(input, target, mask):
  diff = target[mask] - input[mask]
  abs_diff = torch.abs(diff)
  c = torch.max(abs_diff).item() / 5
  leq = (abs_diff <= c).float()
  l2_losses = (diff**2 + c**2) / (2 * c)
  loss = leq * abs_diff + (1 - leq) * l2_losses
  return torch.mean(loss)
  # count = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()
  # masked_loss = loss * mask.float()
  # return torch.mean(torch.sum(masked_loss, dim=[1, 2, 3], keepdim=True) / count)


def MultiscaleBerHuLoss(inputs, target, mask, max_depth=1000):
  loss = 0
  scales = [1 / 4, 1 / 2, 1]
  weights = [0.5, 0.7, 1.0]
  assert (len(inputs) == len(scales))
  for i in range(len(inputs)):
    targetI = F.interpolate(target, scale_factor=scales[i])
    inputI = inputs[i]
    maskI = (targetI > 0) & (targetI <= max_depth) & (~torch.isnan(targetI))
    abs_diff = torch.abs(targetI[maskI] - inputI[maskI])
    c = torch.max(abs_diff).item() / 5
    leq = (abs_diff <= c).float()
    l2_losses = (abs_diff**2 + c**2) / (2 * c)
    loss = weights[i] * torch.mean((leq * abs_diff + (1 - leq) * l2_losses))
  return loss

  # if len(input) == 5:
  #   pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]  # AANet and AANet+
  # elif len(input) == 4:
  #   pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0]
  # elif len(input) == 3:
  #   pyramid_weight = [1.0, 1.0, 1.0]  # 1 scale only
  # elif len(input) == 1:
  #   pyramid_weight = [1.0]  # highest loss only
  # else:
  #   raise NotImplementedError


# def scaleInvariantErrorOri(input, target, mask, lam):
#   D = torch.log(input[mask]) - torch.log(target[mask])
#   n = torch.sum((mask == True))
#   D2 = torch.pow(D, 2)
#   loss = torch.sum(D2) / n - lam * ((torch.sum(D))**2) / (n**2)
#   return loss

# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
# b = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.float32)
# c = np.array([[1, 1, 3], [1, 4, 7], [2, 4, 9]]).astype(np.float32)
# a = torch.from_numpy(a).unsqueeze_(0).unsqueeze_(0)
# b = torch.from_numpy(b).unsqueeze_(0).unsqueeze_(0)
# c = torch.from_numpy(c).unsqueeze_(0).unsqueeze_(0)

# lam = 0.5
# b = b.repeat(2, 1, 1, 1) * 10
# a = torch.cat([a, c], 0) * 10

# mask = b > 0
# print(scaleInvariantLoss(a, b, mask, lam))
# print(scaleInvariantErrorOri(a, b, mask, lam))
# print(F.smooth_l1_loss(a, b))
