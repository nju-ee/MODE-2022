import os
import sys
import torch
import torch.nn as nn
from .basic import SphereConv


# NOTE: Model Initialization methods
def initModelPara(model, initType):
  if initType == None or initType == 'default':
    return
  for m in model.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, SphereConv):
      if initType == 'kaiming_normal':
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
      elif initType == 'xavier_normal':
        nn.init.xavier_normal_(m.weight)
      elif initType == 'kaiming_uniform':
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
      elif initType == 'xavier_uniform':
        nn.init.xavier_uniform_(m.weight)
      elif initType == 'normal':
        nn.init.normal_(m.weight)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
      nn.init.normal_(m.weight, 0, 0.01)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)


def loadStackHourglassOnly(model, savedDictPath):
  pretrained_dict = torch.load(savedDictPath)['state_dict']
  currentDict = model.state_dict()
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in currentDict) and ('feature_extraction' not in k) and ('forfilter1' not in k))}
  currentDict.update(pretrained_dict)
  print("load partial parameter: ")
  model.load_state_dict(currentDict)
  print("loading done!")
