from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataloader import list_deep360_file as lt
from dataloader import MyDeep360Loader as DA
from models import *
from geometry import *
import cv2

parser = argparse.ArgumentParser(description='MODE-Net-stage1')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--dbname', default=None, help='dataset name')
parser.add_argument('--datapath', default='/data/Share/datasets/Deep360/depth_on_1/', help='datapath')
parser.add_argument('--soil', action='store_true', default=False, help='output the intermediate results of soiled dataset')
parser.add_argument('--outpath', default='/home/jinxueqian/MODE-Net_output_stage1/', help='output path')
parser.add_argument('--batch-size', type=int, default=6, help='batch size')
parser.add_argument('--loadmodel', default=None, help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)

if args.dbname == 'Deep360':
  left_img, right_img = lt.listfile_stage1_output(args.datapath, args.soil)
elif args.dbname == '3D60':
  pass
else:
  left_img, right_img, masks = lt.listfile_stage1_output_omnifisheye(args.datapath)

AllImgLoader = torch.utils.data.DataLoader(DA.myDataLoaderStage1Output(left_img, right_img), batch_size=args.batch_size, shuffle=False, num_workers=args.batch_size, drop_last=False)

# Note
# in_height,in_width: shape of input image. using (1024,512) for deep360, (640,320) for fisheye, (512,256) for 3D60
if args.dbname == 'Deep360':
  model = ModesDisparity(args.maxdisp, conv='Sphere', in_height=1024, in_width=512, out_conf=True)
elif args.dbname == '3D60':
  model = ModesDisparity(args.maxdisp, conv='Sphere', in_height=512, in_width=256, out_conf=True)
else:
  model = ModesDisparity(args.maxdisp, conv='Sphere', in_height=640, in_width=320, out_conf=True)

if args.cuda:
  model = nn.DataParallel(model)
  model.cuda()

if args.loadmodel is not None:
  print('Load pretrained model')
  pretrain_dict = torch.load(args.loadmodel)
  model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def output_stage1(imgL, imgR):
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

  if args.dbname != 'Deep360' and args.dbname != '3D60':
    mask_left = masks[cam_pair_dict[cam_pair] * 2]
    mask_right = masks[cam_pair_dict[cam_pair] * 2 + 1]
    conf_map = conf_map * mask_left * mask_right
    depth_l = depth_l * mask_left * mask_right

  if cam_pair == '12':
    return depth_l, conf_map
  elif cam_pair == '13':
    depth_1 = np.expand_dims(depth_l, axis=-1)
    depth_2 = cassini2Cassini(depth_1, 0.5 * math.pi, 0, 0)
    conf_1 = np.expand_dims(conf_map, axis=-1)
    conf_2 = cassini2Cassini(conf_1, 0.5 * math.pi, 0, 0)
    return depth_2[:, :, 0], conf_2[:, :, 0]
  elif cam_pair == '14':
    depth_1 = np.expand_dims(depth_l, axis=-1)
    depth_2 = cassini2Cassini(depth_1, 0.25 * math.pi, 0, 0)
    conf_1 = np.expand_dims(conf_map, axis=-1)
    conf_2 = cassini2Cassini(conf_1, 0.25 * math.pi, 0, 0)
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
  if args.dbname == 'Deep360':
    #------------- Check the output directories -------
    ep_list = os.listdir(args.datapath)  #ep子文件夹名列表
    ep_list.sort()
    for ep in ep_list:
      for subset in ['training', 'validation', 'testing']:
        if not args.soil:
          # disp_pred_path = os.path.join(args.outpath, ep, subset, "disp_pred")
          # if not os.path.exists(disp_pred_path):
          #     os.makedirs(disp_pred_path)
          disp_pred2depth_path = os.path.join(args.outpath, ep, subset, "disp_pred2depth")
          if not os.path.exists(disp_pred2depth_path):
            os.makedirs(disp_pred2depth_path)
          conf_map_path = os.path.join(args.outpath, ep, subset, "conf_map")
          if not os.path.exists(conf_map_path):
            os.makedirs(conf_map_path)
        else:
          # disp_pred_soiled_path = os.path.join(args.outpath, ep, subset, "disp_pred_soiled")
          # if not os.path.exists(disp_pred_soiled_path):
          #     os.makedirs(disp_pred_soiled_path)
          disp_pred2depth_soiled_path = os.path.join(args.outpath, ep, subset, "disp_pred2depth_soiled")
          conf_map_soiled_path = os.path.join(args.outpath, ep, subset, "conf_map_soiled")
          if not os.path.exists(disp_pred2depth_soiled_path):
            os.makedirs(disp_pred2depth_soiled_path)
          if not os.path.exists(conf_map_soiled_path):
            os.makedirs(conf_map_soiled_path)

          if subset == 'testing':
            # for soil_type_dir in ['mud/', 'water/', 'glare/']:
            #     soil_type_path = os.path.join(disp_pred_soiled_path, soil_type_dir)
            #     if not os.path.exists(soil_type_path):
            #         os.makedirs(soil_type_path)

            #     for soil_cam_num_dir in ['1_soiled_cam/', '2_soiled_cam/']:
            #         soil_cam_num_path = os.path.join(soil_type_path, soil_cam_num_dir)
            #         if not os.path.exists(soil_cam_num_path):
            #             os.makedirs(soil_cam_num_path)

            #         for soil_num, soil_num_dir in enumerate(['1_spot/', '2_spot/', '3_spot/', '4_spot/', '5_spot/']):
            #             soil_num_path = os.path.join(soil_cam_num_path, soil_num_dir)
            #             if not os.path.exists(soil_num_path):
            #                 os.makedirs(soil_num_path)

            #             for soil_rate, soil_rate_dir in enumerate(['05percent/', '10percent/', '15percent/', '20percent/']):
            #                 soil_rate_path = os.path.join(soil_num_path, soil_rate_dir)
            #                 if not os.path.exists(soil_rate_path):
            #                     os.makedirs(soil_rate_path)

            for soil_type_dir in ['mud/', 'water/', 'glare/']:
              soil_type_path = os.path.join(disp_pred2depth_soiled_path, soil_type_dir)
              if not os.path.exists(soil_type_path):
                os.makedirs(soil_type_path)

              for soil_cam_num_dir in ['1_soiled_cam/', '2_soiled_cam/']:
                soil_cam_num_path = os.path.join(soil_type_path, soil_cam_num_dir)
                if not os.path.exists(soil_cam_num_path):
                  os.makedirs(soil_cam_num_path)

                for soil_num, soil_num_dir in enumerate(['2_spot/', '3_spot/', '4_spot/', '5_spot/', '6_spot/']):
                  soil_num_path = os.path.join(soil_cam_num_path, soil_num_dir)
                  if not os.path.exists(soil_num_path):
                    os.makedirs(soil_num_path)

                  for soil_rate, soil_rate_dir in enumerate(['05percent/', '10percent/', '15percent/', '20percent/']):
                    soil_rate_path = os.path.join(soil_num_path, soil_rate_dir)
                    if not os.path.exists(soil_rate_path):
                      os.makedirs(soil_rate_path)

            for soil_type_dir in ['mud/', 'water/', 'glare/']:
              soil_type_path = os.path.join(conf_map_soiled_path, soil_type_dir)
              if not os.path.exists(soil_type_path):
                os.makedirs(soil_type_path)

              for soil_cam_num_dir in ['1_soiled_cam/', '2_soiled_cam/']:
                soil_cam_num_path = os.path.join(soil_type_path, soil_cam_num_dir)
                if not os.path.exists(soil_cam_num_path):
                  os.makedirs(soil_cam_num_path)

                for soil_num, soil_num_dir in enumerate(['2_spot/', '3_spot/', '4_spot/', '5_spot/', '6_spot/']):
                  soil_num_path = os.path.join(soil_cam_num_path, soil_num_dir)
                  if not os.path.exists(soil_num_path):
                    os.makedirs(soil_num_path)

                  for soil_rate, soil_rate_dir in enumerate(['05percent/', '10percent/', '15percent/', '20percent/']):
                    soil_rate_path = os.path.join(soil_num_path, soil_rate_dir)
                    if not os.path.exists(soil_rate_path):
                      os.makedirs(soil_rate_path)
    #----------------------------------------------------------
  elif args.dbname == '3D60':
    pass
  else:
    #------------- Check the output directories -------
    for subset in ['training', 'testing']:
      # disp_pred_path = os.path.join(args.outpath, ep, subset, "disp_pred")
      # if not os.path.exists(disp_pred_path):
      #     os.makedirs(disp_pred_path)
      disp_pred2depth_path = os.path.join(args.outpath, subset, "disp_pred2depth")
      if not os.path.exists(disp_pred2depth_path):
        os.makedirs(disp_pred2depth_path)
      conf_map_path = os.path.join(args.outpath, subset, "conf_map")
      if not os.path.exists(conf_map_path):
        os.makedirs(conf_map_path)
    #----------------------------------------------------------

  for batch_idx, (left_name, imgL, imgR) in enumerate(AllImgLoader):
    print("\rStage1 输出进度： {:.2f}%".format(100 * (batch_idx + 1) / len(AllImgLoader)), end='')

    pred_disp_batch, conf_map_batch = output_stage1(imgL, imgR)

    for i in range(pred_disp_batch.shape[0]):
      outpath = left_name[i].replace(args.datapath, args.outpath)
      outpath = outpath[:-8]
      #------------- save disp_pred ------------------
      # outpath_disp = outpath.replace("rgb","disp_pred")
      # np.save(outpath_disp + "disp_pred.npy", pred_disp_batch[i])
      #------------- do disp2depth ------------------
      depth_at_1, conf_at_1 = disp2depth(pred_disp_batch[i], conf_map_batch[i], left_name[i][-11:-9])
      outpath_depth = outpath.replace("rgb", "disp_pred2depth")
      np.save(outpath_depth + "disp_pred2depth.npy", depth_at_1)
      #------------- save conf_map ------------------
      outpath_conf = outpath.replace("rgb", "conf_map")
      cv2.imwrite(outpath_conf + "conf_map.png", conf_at_1 * 255)


if __name__ == '__main__':
  main()
