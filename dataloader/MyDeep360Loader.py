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

class myDataLoaderStage1(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]


        left_img = self.loader(left)
        right_img = self.loader(right)
        disp_map = self.dploader(disp_L)
        disp_map = np.ascontiguousarray(disp_map,dtype=np.float32)

        if self.training:  
            w, h = left_img.size
            th, tw = 512, 256

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            disp_map = disp_map[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform_stage1(augment=False)  
            left_img   = processed(left_img)
            right_img  = processed(right_img)

            return left_img, right_img, disp_map
        else:
            processed = preprocess.get_transform_stage1(augment=False)  
            left_img       = processed(left_img)
            right_img      = processed(right_img) 
            return left_img, right_img, disp_map

    def __len__(self):
        return len(self.left)


class myDataLoaderStage1Output(data.Dataset):
    def __init__(self, left, right, input_resize, loader=default_loader):
 
        self.left = left
        self.right = right
        self.input_resize = input_resize
        self.loader = loader

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]

        left_img = self.loader(left)
        right_img = self.loader(right)

        left_img = left_img.resize((self.input_resize[1], self.input_resize[0]))
        right_img = right_img.resize((self.input_resize[1], self.input_resize[0]))

        processed = preprocess.get_transform_stage1(augment=False)  
        left_img       = processed(left_img)
        right_img      = processed(right_img) 
        return left, left_img, right_img

    def __len__(self):
        return len(self.left)

class myDataLoaderRearview(data.Dataset):
    def __init__(self, left, right, input_resize, loader=default_loader):
 
        self.left = left
        self.right = right
        self.input_resize = input_resize
        self.loader = loader

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]

        left_img = self.loader(left)
        right_img = self.loader(right)

        left_img = left_img.transpose(Image.ROTATE_270)
        right_img = right_img.transpose(Image.ROTATE_270)

        left_img = left_img.resize((self.input_resize[1], self.input_resize[0]))
        right_img = right_img.resize((self.input_resize[1], self.input_resize[0]))

        processed = preprocess.get_transform_stage1(augment=False)  
        left_img       = processed(left_img)
        right_img      = processed(right_img) 
        return left, left_img, right_img

    def __len__(self):
        return len(self.left)


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
