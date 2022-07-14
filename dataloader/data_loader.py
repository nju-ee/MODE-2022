import enum
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
from . import preprocess 
import numpy as np
import cv2

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return np.load(path).astype(np.float32)

def depth_loader(path):
    depth = np.load(path).astype(np.float32)
    return np.expand_dims(depth, axis=-1)

def conf_loader(path):
    conf = cv2.imread(path)
    return np.expand_dims((conf[:,:,0]/255.0).astype(np.float32), axis=0)

class myDataLoaderStage2(data.Dataset):
    def __init__(self, depthes, confs, rgbs, gt, resize, training, depthloader= depth_loader, rgbloader=default_loader):
 
        self.depthes = depthes
        self.confs = confs
        self.rgbs = rgbs
        self.gt = gt
        self.depthloader = depthloader
        self.rgbloader = rgbloader
        self.resize = resize
        self.training = training

    def __getitem__(self, index):
        depthes = []
        confs = []
        rgbs = []

        for depth in self.depthes:
            depthes.append(self.depthloader(depth[index]))
        for conf in self.confs:
            confs.append(conf_loader(conf[index]))
        for rgb in self.rgbs:
            rgbs.append(self.rgbloader(rgb[index]))
        gt = self.depthloader(self.gt[index])
        gt = np.squeeze(gt, axis=-1)
        gt = np.ascontiguousarray(gt,dtype=np.float32)

        if self.resize:
            for i, depth in enumerate(depthes):
                depthes[i] = depth[::2, ::2, :]
            for i, conf in enumerate(confs):
                confs[i] = conf[:, ::2, ::2]
            w, h = rgbs[0].size
            for i, rgb in enumerate(rgbs):
                rgbs[i] = rgb.resize((int(w/2), int(h/2)))
            if self.training:
                gt = gt[::2, ::2]

        processed = preprocess.get_transform_stage2(augment=False)  
        for i, depth in enumerate(depthes):
            depthes[i] = processed(depth)

        processed = preprocess.get_transform_stage1(augment=False)
        for i, rgb in enumerate(rgbs):
            rgbs[i] = processed(rgb)

        return self.gt[index], depthes, confs, rgbs, gt

    def __len__(self):
        return len(self.depthes[0])
