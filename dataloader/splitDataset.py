import os
import sys
import random
"""
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
     - rgb
     - disp
     - depth

"""


def splitMulti360Dataset():
  rate = [7, 1, 2]
  total = sum(rate)
  rootDir = '/home/liming/projects/datasets/Deep360/depth_on_1'
  prefix = 'deep360'
  subDirList = os.listdir(rootDir)
  fileNameSet = set()
  for subDir in subDirList:
    curDir = os.path.join(rootDir, subDir, 'rgb')
    fileList = os.listdir(curDir)
    for fn in fileList:
      fileNameSet.add(os.path.join(subDir, 'rgb', fn.split('_')[0]))
  totalFile = len(fileNameSet)
  print("Num of All scenes: {}".format(totalFile))
  fileNameList = list(fileNameSet)
  random.shuffle(fileNameList)
  tr_va = totalFile // total * rate[0]
  if totalFile % total > 0: tr_va = tr_va + 1
  va_te = totalFile // total * (rate[0] + rate[1])
  trainList = fileNameList[0:tr_va]
  valList = fileNameList[tr_va:va_te]
  testList = fileNameList[va_te:]
  print("train: {}, val: {}, test: {}".format(len(trainList), len(valList), len(testList)))

  # save to txt
  # all
  fileNameList = [line + '\n' for line in fileNameList]
  with open('{}_all.txt'.format(prefix), 'w') as f:
    f.writelines(fileNameList)
  # train
  trainList = [line + '\n' for line in trainList]
  with open('{}_train.txt'.format(prefix), 'w') as f:
    f.writelines(trainList)
  # validation
  valList = [line + '\n' for line in valList]
  with open('{}_val.txt'.format(prefix), 'w') as f:
    f.writelines(valList)
  # train
  testList = [line + '\n' for line in testList]
  with open('{}_test.txt'.format(prefix), 'w') as f:
    f.writelines(testList)


if __name__ == '__main__':
  splitMulti360Dataset()

  # rate = [7, 1, 2]  #train:val:test
  # total = rate[0] + rate[1] + rate[2]
  # rootDir = "../../../datasets/360depth"
  # subDirList = os.listdir(rootDir)
  # fileNameSet = set()
  # for subDir in subDirList:
  #   curDir = os.path.join(rootDir, subDir, 'rgb_png')
  #   fileList = os.listdir(curDir)
  #   for fn in fileList:
  #     fileNameSet.add(os.path.join(subDir, 'rgb_png', fn.split('_')[0]))
  # totalFile = len(fileNameSet)
  # print(totalFile)
  # fileNameList = list(fileNameSet)
  # random.shuffle(fileNameList)
  # tr_va = totalFile // total * rate[0]
  # if totalFile % total > 0: tr_va = tr_va + 1
  # va_te = totalFile // total * (rate[0] + rate[1])
  # trainList = fileNameList[0:tr_va]
  # valList = fileNameList[tr_va:va_te]
  # testList = fileNameList[va_te:]

  # # save to txt
  # # train
  # trainList = [line + '\n' for line in trainList]
  # with open('mulit_train.txt', 'w') as f:
  #   f.writelines(trainList)
  # # validation
  # valList = [line + '\n' for line in valList]
  # with open('mulit_val.txt', 'w') as f:
  #   f.writelines(valList)
  # # train
  # testList = [line + '\n' for line in testList]
  # with open('mulit_test.txt', 'w') as f:
  #   f.writelines(testList)
