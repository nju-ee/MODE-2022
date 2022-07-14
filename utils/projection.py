import os
import numpy as np
import pickle

import torch


def batchERP(inputTensor, meshLevel, interMap, outSize=(256, 512), dense=True):
  """
  return: ERP depth map interpolate from s2 depth signal, with a shpae of NxCxHxW

  inputTensor: batch*channel*num tensor, the output of spherical cnns

  meshLevel: integer in [0,7] the level of mesh

  interMap: 1d tensor to indentify the interpolation index

  outSize: tuple, output image size, H,W for the 4-dim tensor(keep N,C as input)

  dense: bool, dense or sparse interpolation
  """
  if dense:
    b, ch, n = inputTensor.shape
    assert (ch == 1 or ch == 3 or ch == 4)
    outS = [b, ch, outSize[0] * outSize[1]]
    # img = torch.zeros(outS).cuda()
    inter = interMap.repeat(b, ch, 1)
    img = torch.gather(inputTensor, 2, inter)
    img = img.view([b, ch, outSize[0], outSize[1]]).cuda()
    return img

  else:
    b, ch, n = inputTensor.shape
    assert (ch == 1 or ch == 3 or ch == 4)
    outS = [b, ch, outSize[0] * outSize[1]]
    # img = torch.zeros(outS).cuda()
    inter = interMap.repeat(b, ch, 1)
    img = img.scatter(2, inter, inputTensor)
    img = img.view([b, ch, outSize[0], outSize[1]]).cuda()
    return img


def verToHorMap(height, width, minTheta=-np.pi / 2, minPhi=-3 * np.pi / 2):
  u = np.expand_dims(np.linspace(0, width - 1, width), 0)
  u = np.repeat(u, height, 0)
  phi = u / height * np.pi + minPhi
  back = (phi < minPhi + np.pi / 2) | (phi > minPhi + 3 * np.pi / 2)
  v = np.expand_dims(np.linspace(0, height - 1, height), 0).T
  v = np.repeat(v, width, 1)
  theta0 = v / height * np.pi + minTheta
  theta = np.arctan(-np.sin(phi) * np.tan(theta0))
  thetaInv = np.arctan(-np.tan(theta0) / (np.sin(phi) + 1e-10))
  vTh = ((theta - minTheta) / np.pi * (height - 1)).astype(np.int)
  hTv = ((thetaInv - minTheta) / np.pi * (height - 1)).astype(np.int)
  return vTh, hTv


def batchHoriPro(inputTensor, vthMap):
  b, c, h, w = inputTensor.shape
  vthMap = vthMap.repeat(b, c, 1, 1)
  #print(vthMap.shape, inputTensor.shape)
  pro = torch.gather(inputTensor, 2, vthMap)
  out = torch.cat([torch.flip(pro[:, :, :, 0:w // 4], (3,)), torch.flip(pro[:, :, :, 3 * w // 4:], (3,))], dim=3)
  out = torch.cat([pro[:, :, :, w // 4:3 * w // 4], out], dim=2)
  #print(out.shape)
  return out


def batchVerPro(inputTensor, htvMap):
  b, c, h, w = inputTensor.shape
  htvMap = htvMap.repeat(b, c, 1, 1)
  #print(vthMap.shape, inputTensor.shape)
  input = torch.cat([torch.flip(inputTensor[:, :, h // 2:, 0:w // 2], (3,)), inputTensor[:, :, 0:h // 2, :], torch.flip(inputTensor[:, :, h // 2:, w // 2:], (3,))], dim=3)
  out = torch.gather(input, 2, htvMap)
  #print(out.shape)
  return out


def generateDepthDispMap(short, long, baseline):
  theta2 = np.expand_dims(np.linspace(0, short - 1, short), 0)
  theta2 = np.repeat(theta2, long, 0)
  theta2 = theta2 / 255 * np.pi + (-np.pi / 2)
  cosTheta2 = np.cos(theta2)
  cosAngleCoeMap = (cosTheta2 * baseline * short / np.pi).astype(np.float32)
  return cosAngleCoeMap


def batchDepthToDisp(depth, cosAngleCoeMap):
  # cosAngleCoeMap = cos(theta)*baseline*shortEdgePixels/pi
  b, c, h, w = depth.shape
  cosAngleCoeMap = cosAngleCoeMap.repeat(b, c, 1, 1)
  disp = cosAngleCoeMap / depth
  return disp


def batchDispToDepth(disp, cosAngleCoeMap):
  # cosAngleCoeMap = cos(theta)*baseline*shortEdgePixels/pi
  b, c, h, w = disp.shape
  cosAngleCoeMap = cosAngleCoeMap.repeat(b, c, 1, 1)
  depth = cosAngleCoeMap / disp
  return depth
