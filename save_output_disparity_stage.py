from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataloader import list_deep360_disparity_train, list_deep360_disparity_test
from dataloader import Deep360DatasetDisparity
from models import ModeDisparity
from utils.geometry import rotateCassini, depthViewTransWithConf
import cv2

parser = argparse.ArgumentParser(description='MODE - save disparity and confidence outputs')
parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--dbname', default='Deep360', help='dataset name')
parser.add_argument('--datapath', default='../../datasets/Deep360/', help='datapath')
parser.add_argument('--soiled', action='store_true', default=False, help='output the intermediate results of soiled dataset')
parser.add_argument('--outpath', default='./outputs/Deep360PredDepth/', help='output path')
parser.add_argument('--batch_size', type=int, default=6, help='batch size')
parser.add_argument('--checkpoint_disp', default=None, help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)

if args.dbname == 'Deep360':
  train_left_img, train_right_img, train_left_disp, val_left_img, val_right_img, val_left_disp = list_deep360_disparity_train(args.datapath, args.soiled)
  test_left_img, test_right_img, test_left_disp = list_deep360_disparity_test(args.datapath, args.soiled)
  total_num = (len(train_left_img) + len(val_left_img) + len(test_left_img))
  # list all files---------------------------------
  # left
  all_left = train_left_img
  all_left.extend(val_left_img)
  all_left.extend(test_left_img)
  # right
  all_right = train_right_img
  all_right.extend(val_right_img)
  all_right.extend(test_right_img)
  # disp
  all_disp = train_left_disp
  all_disp.extend(val_left_disp)
  all_disp.extend(test_left_disp)
  assert (len(all_left) == len(all_right) == len(all_disp) == total_num)
  #------------------------------------------------

AllImgLoader = torch.utils.data.DataLoader(Deep360DatasetDisparity(all_left, all_right, all_disp), batch_size=args.batch_size, shuffle=False, num_workers=args.batch_size, drop_last=False)

# Note
# in_height,in_width: shape of input image. using (1024,512) for deep360
if args.dbname == 'Deep360':
  model = ModeDisparity(args.max_disp, conv='Sphere', in_height=1024, in_width=512, out_conf=True)

if args.cuda:
  model = nn.DataParallel(model)
  model.cuda()

if args.checkpoint_disp is not None:
  print('Load pretrained model')
  pretrain_dict = torch.load(args.checkpoint_disp)
  model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def output_disp_and_conf(imgL, imgR):
  model.eval()

  if args.cuda:
    imgL, imgR = imgL.cuda(), imgR.cuda()

  if imgL.shape[2] % 16 != 0:
    times = imgL.shape[2] // 16
    top_pad = (times + 1) * 16 - imgL.shape[2]
  else:
    top_pad = 0

  if imgL.shape[3] % 16 != 0:
    times = imgL.shape[3] // 16
    right_pad = (times + 1) * 16 - imgL.shape[3]
  else:
    right_pad = 0

  imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
  imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

  with torch.no_grad():
    output3, conf_map = model(imgL, imgR)
    output3 = torch.squeeze(output3, 1)
    conf_map = torch.squeeze(conf_map, 1)

  if top_pad != 0:
    output3 = output3[:, top_pad:, :]
  if right_pad != 0:
    output3 = output3[:, :, :-right_pad]

  return output3.data.cpu().numpy(), conf_map.data.cpu().numpy()


def disp2depth(disp, conf_map, cam_pair):
  cam_pair_dict = {'12': 0, '13': 1, '14': 2, '23': 3, '24': 4, '34': 5}

  if args.dbname == 'Deep360':
    baseline = np.array([1, 1, math.sqrt(2), math.sqrt(2), 1, 1]).astype(np.float32)
  elif args.dbname == '3D60':
    pass
  else:
    baseline = np.array([0.6 * math.sqrt(2), 0.6 * math.sqrt(2), 1.2, 1.2, 0.6 * math.sqrt(2), 0.6 * math.sqrt(2)]).astype(np.float32)

  output_h = disp.shape[0]
  output_w = disp.shape[1]

  phi_l_start = 0.5 * math.pi - (0.5 * math.pi / output_w)
  phi_l_end = -0.5 * math.pi
  phi_l_step = math.pi / output_w
  phi_l_range = np.arange(phi_l_start, phi_l_end, -phi_l_step)
  phi_l_map = np.array([phi_l_range for j in range(output_h)]).astype(np.float32)

  mask_disp_is_0 = disp == 0
  disp_not_0 = np.ma.array(disp, mask=mask_disp_is_0)

  phi_r_map = disp_not_0 * math.pi / output_w + phi_l_map

  # sin theory
  depth_l = baseline[cam_pair_dict[cam_pair]] * np.sin(math.pi / 2 - phi_r_map) / np.sin(phi_r_map - phi_l_map)
  depth_l = depth_l.filled(1000)
  depth_l[depth_l > 1000] = 1000
  depth_l[depth_l < 0] = 0

  if cam_pair == '12':
    return depth_l, conf_map
  elif cam_pair == '13':
    depth_1 = np.expand_dims(depth_l, axis=-1)
    depth_2 = rotateCassini(depth_1, 0.5 * math.pi, 0, 0)
    conf_1 = np.expand_dims(conf_map, axis=-1)
    conf_2 = rotateCassini(conf_1, 0.5 * math.pi, 0, 0)
    return depth_2[:, :, 0], conf_2[:, :, 0]
  elif cam_pair == '14':
    depth_1 = np.expand_dims(depth_l, axis=-1)
    depth_2 = rotateCassini(depth_1, 0.25 * math.pi, 0, 0)
    conf_1 = np.expand_dims(conf_map, axis=-1)
    conf_2 = rotateCassini(conf_1, 0.25 * math.pi, 0, 0)
    return depth_2[:, :, 0], conf_2[:, :, 0]
  elif cam_pair == '23':
    depth_2, conf_2 = depthViewTransWithConf(depth_l, conf_map, 0, -math.sqrt(2) / 2, -math.sqrt(2) / 2, 0.75 * math.pi, 0, 0)
    return depth_2, conf_2
  elif cam_pair == '24':
    depth_2, conf_2 = depthViewTransWithConf(depth_l, conf_map, 0, -1, 0, 0.5 * math.pi, 0, 0)
    return depth_2, conf_2
  elif cam_pair == '34':
    depth_2, conf_2 = depthViewTransWithConf(depth_l, conf_map, 0, 1, 0, 0, 0, 0)
    return depth_2, conf_2
  else:
    print("Error! Wrong Cam_pair!")
    return None


def main():
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if args.cuda else "cpu")
  if args.dbname == 'Deep360':
    #------------- Check the output directories -------
    ep_list = ["ep%d_500frames" % i for i in range(1, 7)]
    ep_list.sort()
    outdir_name = "disp_pred2depth" if not args.soiled else "disp_pred2depth_soiled"
    outdir_conf_name = "conf_map" if not args.soiled else "conf_map_soiled"
    for ep in ep_list:
      for subset in ['training', 'validation', 'testing']:
        disp_pred2depth_path = os.path.join(args.outpath, ep, subset, outdir_name)
        os.makedirs(disp_pred2depth_path, exist_ok=True)
        conf_map_path = os.path.join(args.outpath, ep, subset, outdir_conf_name)
        os.makedirs(conf_map_path, exist_ok=True)
    #----------------------------------------------------------
  for batchIdx, batchData in enumerate(AllImgLoader):
    print("\rDisparity output progress: {:.2f}%".format(100 * (batchIdx + 1) / len(AllImgLoader)), end='')
    leftImg = batchData['leftImg'].to(device)
    rightImg = batchData['rightImg'].to(device)
    dispName = batchData['dispNames']

    pred_disp_batch, conf_map_batch = output_disp_and_conf(leftImg, rightImg)

    for i in range(pred_disp_batch.shape[0]):
      outpath = dispName[i].replace(args.datapath, args.outpath)
      outpath = outpath[:-8]
      #------------- save disp_pred ------------------
      # outpath_disp = outpath.replace("rgb","disp_pred")
      # np.save(outpath_disp + "disp_pred.npy", pred_disp_batch[i])
      #------------- do disp2depth ------------------
      depth_at_1, conf_at_1 = disp2depth(pred_disp_batch[i], conf_map_batch[i], dispName[i][-11:-9])
      outpath_depth = outpath.replace("disp", outdir_name)
      np.savez(outpath_depth + "disp_pred2depth.npz", depth_at_1)  #save npz files
      #------------- save conf_map ------------------
      outpath_conf = outpath.replace("disp", outdir_conf_name)
      cv2.imwrite(outpath_conf + "conf_map.png", conf_at_1 * 255)


if __name__ == '__main__':
  main()
