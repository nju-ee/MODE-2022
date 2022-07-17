import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import random
import numpy as np
import math

__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175,
                            0.0188,
                            0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,
         0.7192,
         0.4009],
        [-0.5808,
         -0.0045,
         -0.8140],
        [-0.5836,
         -0.6948,
         0.4203],
    ])
}

__deep360_stats = {'mean': [0], 'std': [1]}


def inception_preproccess(input_size, normalize=__imagenet_stats):
  return transforms.Compose([transforms.RandomSizedCrop(input_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(**normalize)])


def inception_color_preproccess(normalize=__imagenet_stats):
  return transforms.Compose([
      transforms.ToTensor(),
      transforms.ColorJitter(
          brightness=0.4,
          contrast=0.4,
          saturation=0.4,
      ),
      Lighting(0.1,
               __imagenet_pca['eigval'],
               __imagenet_pca['eigvec']),
      transforms.Normalize(**normalize)
  ])


def color_normalize(normalize=__imagenet_stats):
  t_list = [
      transforms.ToTensor(),
      transforms.Normalize(**normalize),
  ]
  return transforms.Compose(t_list)


def depth_normalize(normalize=__deep360_stats):
  t_list = [
      transforms.ToTensor(),
      transforms.Normalize(**normalize),
  ]
  return transforms.Compose(t_list)


def get_transform_stage1(name='imagenet', normalize=None, augment=True):
  normalize = __imagenet_stats
  if augment:
    return inception_color_preproccess(normalize=normalize)
  else:
    return color_normalize(normalize=normalize)


def get_transform_stage2(name='deep360', normalize=None, augment=False):
  normalize = __deep360_stats
  return depth_normalize(normalize=normalize)


class Lighting(object):
  """Lighting noise(AlexNet - style PCA - based noise)"""
  def __init__(self, alphastd, eigval, eigvec):
    self.alphastd = alphastd
    self.eigval = eigval
    self.eigvec = eigvec

  def __call__(self, img):
    if self.alphastd == 0:
      return img

    alpha = img.new().resize_(3).normal_(0, self.alphastd)
    rgb = self.eigvec.type_as(img).clone()\
        .mul(alpha.view(1, 3).expand(3, 3))\
        .mul(self.eigval.view(1, 3).expand(3, 3))\
        .sum(1).squeeze()

    return img.add(rgb.view(3, 1, 1).expand_as(img))
