import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
from . import preprocess 
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return np.load(path).astype(np.float32)

def depth_loader(path):
    return np.load(path).astype(np.float32)

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

            processed = preprocess.get_transform(augment=False)  
            left_img   = processed(left_img)
            right_img  = processed(right_img)

            return left_img, right_img, disp_map
        else:
            processed = preprocess.get_transform(augment=False)  
            left_img       = processed(left_img)
            right_img      = processed(right_img) 
            return left_img, right_img, disp_map

    def __len__(self):
        return len(self.left)


class myDataLoaderStage1Output(data.Dataset):
    def __init__(self, left, right, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.loader = loader
        self.dploader = dploader

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]

        left_img = self.loader(left)
        right_img = self.loader(right)

        processed = preprocess.get_transform(augment=False)  
        left_img       = processed(left_img)
        right_img      = processed(right_img) 
        return left, left_img, right_img

    def __len__(self):
        return len(self.left)


class myDataLoaderStage2(data.Dataset):
    def __init__(self, input12, input13, input14, input23, input24, input34, depth, training, depthloader= depth_loader):
 
        self.input12 = input12
        self.input13 = input13
        self.input14 = input14
        self.input23 = input23
        self.input24 = input24
        self.input34 = input34
        self.depth = depth
        self.depthloader = depthloader
        self.training = training

    def __getitem__(self, index):
        input12_path  = self.input12[index]
        input13_path  = self.input13[index]
        input14_path  = self.input14[index]
        input23_path  = self.input23[index]
        input24_path  = self.input24[index]
        input34_path  = self.input34[index]
        depth_path = self.depth[index]

        input12 = self.depthloader(input12_path)
        input13 = self.depthloader(input13_path)
        input14 = self.depthloader(input14_path)
        input23 = self.depthloader(input23_path)
        input24 = self.depthloader(input24_path)
        input34 = self.depthloader(input34_path)
        depth = self.depthloader(depth_path)
        depth = np.ascontiguousarray(depth,dtype=np.float32)

        if self.training:  
            # h, w = input12.shape
            # th, tw = 512, 256

            # x1 = random.randint(0, w - tw)
            # y1 = random.randint(0, h - th)

            # input12 = input12[y1:y1 + th, x1:x1 + tw]
            # input13 = input13[y1:y1 + th, x1:x1 + tw]
            # input14 = input14[y1:y1 + th, x1:x1 + tw]
            # input23 = input23[y1:y1 + th, x1:x1 + tw]
            # input24 = input24[y1:y1 + th, x1:x1 + tw]
            # input34 = input34[y1:y1 + th, x1:x1 + tw]
            # depth = depth[y1:y1 + th, x1:x1 + tw]

            return input12, input13, input14, input23, input24, input34, depth
        else:
            return input12, input13, input14, input23, input24, input34, depth

    def __len__(self):
        return len(self.input12)