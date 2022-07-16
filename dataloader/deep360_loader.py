import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import random
from PIL import Image
import numpy as np
import cv2

if __name__ == '__main__':
  import preprocess
else:
  from . import preprocess


def default_loader(path):
  return Image.open(path).convert('RGB')


def disparity_loader(path):
  return np.load(path)['arr_0'].astype(np.float32)


def depth_loader(path):
  depth = np.load(path)['arr_0'].astype(np.float32)
  return np.expand_dims(depth, axis=-1)


def conf_loader(path):
  conf = cv2.imread(path)
  return np.expand_dims((conf[:, :, 0] / 255.0).astype(np.float32), axis=0)


"""
NOTE:

Multi view 360 depth Dataset
3000 scene points
each point contains 6 pairs of left-right cassini preojection images
(12,13,14,23,24,34), 6 disparity maps and 1 depth maps on view 1

Dataset directory structure:
Deep360
 - depth_on_1
   - ep1_500frames
   - ep2_500frames
   - ep3_500frames
   - ep4_500frames
   - ep5_500frames
   - ep6_500frames
     - training
     - testing
     - validation
       - rgb
       - disp
       - depth
       - rgb_soiled

"""


class Deep360DatasetDisparity(Dataset):
  def __init__(self, leftImgs, rightImgs, disps, shape=(1024, 512), crop=False, disploader=disparity_loader, rgbloader=default_loader):
    # NOTE：Arguments
    # leftImgs: a list contains file names of left images
    # rightImgs: a list contains file names of right images
    # disps: a list contains file names of groundtruth disparity maps
    # shape: img shape of input data. a tuple of (height, width)
    # crop: Bool. if set True, will use random crop
    # disploader: function to load dispairity maps
    # rgbloader: function to load left and right RGB images

    # Initialization
    super(Deep360DatasetDisparity, self).__init__()

    # Member variable assignment
    self.crop = crop
    self.height, self.width = shape
    self.processed = preprocess.get_transform(augment=False)  # transform of rgb images
    self.leftImgs = leftImgs
    self.rightImgs = rightImgs
    self.disps = disps
    self.disp_loader = disploader
    self.rgb_loader = rgbloader

  def __getitem__(self, index):
    inputs = {}
    leftName = self.leftImgs[index]
    rightName = self.rightImgs[index]
    dispName = self.disps[index]
    left = self.rgb_loader(leftName)
    right = self.rgb_loader(rightName)
    disp = self.disp_loader(dispName)
    # if need to resize
    w, h = left.size
    if (w != self.width):
      left = left.resize((self.width, self.height))
      right = right.resize((self.width, self.height))
      disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_NEAREST) * (self.width / w)
    # if random crop
    if self.crop:
      w, h = left.size
      th, tw = 512, 256
      x1 = random.randint(0, w - tw)
      y1 = random.randint(0, h - th)
      leftImg = leftImg.crop((x1, y1, x1 + tw, y1 + th))
      rightImg = rightImg.crop((x1, y1, x1 + tw, y1 + th))
      dispMap = dispMap[y1:y1 + th, x1:x1 + tw]
    disp = np.ascontiguousarray(disp, dtype=np.float32)
    left = self.processed(left)
    right = self.processed(right)
    disp = torch.from_numpy(disp).unsqueeze_(0)
    inputs = {'leftImg': left, 'rightImg': right, 'dispMap': disp, 'dispNames': dispName}
    return inputs

  def __len__(self):
    return len(self.disps)


if __name__ == '__main__':
  from list_deep360_file import listfile_disparity_train, listfile_disparity_test
  train_left_img, train_right_img, train_left_disp, val_left_img, val_right_img, val_left_disp = listfile_disparity_train('../../../datasets/MODE_Datasets/Deep360/', soil=True)
  test_left_img, test_right_img, test_left_disp = listfile_disparity_test('../../../datasets/MODE_Datasets/Deep360/', soil=True)
  all_left = train_left_img
  all_left.extend(val_left_img)
  all_left.extend(test_left_img)
  print(len(all_left))
  da = Deep360DatasetDisparity(leftImgs=train_left_img, rightImgs=train_right_img, disps=train_left_disp)
  da_vali = Deep360DatasetDisparity(leftImgs=val_left_img, rightImgs=val_right_img, disps=val_left_disp)
  da_test = Deep360DatasetDisparity(leftImgs=test_left_img, rightImgs=test_right_img, disps=test_left_disp)
  print(len(da), len(da_vali), len(da_test))
  myDL = torch.utils.data.DataLoader(da_test, batch_size=1, num_workers=1, pin_memory=False, shuffle=True)
  for id, batch in enumerate(myDL):
    # test output inter fusion
    print(batch['leftImg'].shape)
    print(batch['rightImg'].shape)
    print(batch['dispMap'].shape)
    print(batch['dispNames'])
