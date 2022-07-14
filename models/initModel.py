import os
import sys
import torch
import torch.nn as nn
from .basic import SphereConv


# NOTE: Model Initialization methods
def initModel(model, initType):
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


def loadRegressionOnly(model, savedDictPath):
  pretrained_dict = torch.load(savedDictPath)['state_dict']
  currentDict = model.state_dict()
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in currentDict) and ('feature_extraction' not in k) and ('forfilter1' not in k))}
  currentDict.update(pretrained_dict)
  print("load partial parameter: ")
  model.load_state_dict(currentDict)
  print("loading done!")


def freezeSHG(model):
  layerNamesParallel = [model.module.dres0, model.module.dres1, model.module.dres2, model.module.dres3, model.module.dres4, model.module.classif1, model.module.classif2, model.module.classif3]
  for layer in layerNamesParallel:
    for param in layer.parameters():
      param.requires_grad = False


def unfreezeSHG(model):
  #layerNames = [model.dres0, model.dres1, model.dres2, model.dres3, model.dres4, model.classif1, model.classif2, model.classif3]
  layerNamesParallel = [model.module.dres0, model.module.dres1, model.module.dres2, model.module.dres3, model.module.dres4, model.module.classif1, model.module.classif2, model.module.classif3]
  for layer in layerNamesParallel:
    for param in layer.parameters():
      param.requires_grad = True
