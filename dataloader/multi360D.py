###################################
# multi-view 360 dataset pytorch dataloader
###################################
import os
import sys
import pickle

import numpy as np
import cv2
import PIL.Image as Image
import datetime
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

############################################################################################################
# We use a text file to hold our dataset's filenames

#############################################################################################################


class Multi360DepthData(Dataset):
  #360D Dataset#
  def __init__(self, filenamesFile, configFile, delimiter, numCamera=4, inputShape=(512, 1024), transform=None, rescaled=False, dataType='ca'):
    #########################################################################################################
    # Arguments:
    # -filenamesFile: Absolute path to the aforementioned filenames .txt file
    # -delimiter    : delimiter in filenames file
    # -numCamera    : num of Cameras
    # -inputShape   : input image shape, a tuple consists of (height,width)
    # -transform    : (Optional) transform to be applied on a sample
    # -rescaled     : bool, rescaled or not
    # -dataType     : type of input imgs. 'erp' = Equirectangular projection, 'ca' = Cassini projection
    #########################################################################################################
    self.height = inputShape[0]
    self.width = inputShape[1]
    self.datasetRoot = '../../datasets/3D60/'
    # meshFile = pickle.load(open('./meshfiles/icosphere_{}.pkl'.format(meshLevel), 'rb'))
    # self.V = meshFile['V']
    self.sample = {}  # one dataset sample (dictionary)
    self.resize2 = transforms.Resize([self.height // 2, self.width // 2])  # function to resize input image by a factor of 2
    self.resize4 = transforms.Resize([self.height // 4, self.width // 4])  # function to resize input image by a factor of 4
    self.pilToTensor = transforms.ToTensor() if transform is None else transforms.Compose(([
        transform,
        transforms.ToTensor()  # function to convert pillow image to tensor
    ]))
    self.filenamesFilePath = filenamesFile  # file containing image paths to load
    self.configPath = configFile
    self.delimiter = delimiter  # delimiter in filenames file
    self.loadConfig()
    with open(self.filenamesFilePath) as f:
      self.fileList = f.readlines()
    self.length = len(self.fileList)

  def loadConfig(self):
    with open(self.configPath) as f:
      data = json.load(f)
      assert data["datasetName"] == "maulti360Depth"
      self.numCamera = data["numCamera"]
      self.numPairs = data["numPairs"]
      self.imgPairs = data["imgPairs"]

  def genPairsName(self, l, r):
    leftName = l + r + "_rgb" + l + ".png"
    rightName = l + r + "_rgb" + r + ".png"
    dispName = l + r + "_disp" + l + ".npy"
    return leftName, rightName, dispName

  def loadItem(self, idx):
    prefix = self.fileList[idx]
    subDir = prefix.split('/')[0]
    preName = prefix.split('/')[2]
    item = {}
    rgb = []
    dispL = []
    for rgbP in self.imgPairs:
      l, r, d = self.genPairsName(rgbP[0], rgbP[1])
      lImg = self.pilToTensor(Image.open(os.path.join(self.datasetRoot, prefix + l)))
      rImg = self.pilToTensor(Image.open(os.path.join(self.datasetRoot, prefix + r)))
      rgb.append([lImg, rImg])
      dname = subDir + "/disp_npy/" + preName + d
      disp = torch.from_numpy(np.load(os.path.join(self.datasetRoot, dname))).unsqueeze_(0)
      dispL.append(disp)
    depthName = subDir + "/depth_npy/" + preName + "_depth.npy"
    depth = torch.from_numpy(np.load(os.path.join(self.datasetRoot, depthName))).unsqueeze_(0)
    item['rgbImagePairs'] = rgb
    item['dispLeft'] = dispL
    item['depth'] = depth
    return item

  # torch override
  # returns samples length
  def __len__(self):
    return self.length

  # torch override
  def __getitem__(self, idx):
    return self.loadItem(idx)


if __name__ == "__main__":
  mydataset = Multi360DepthData("xxx.txt", " ")
