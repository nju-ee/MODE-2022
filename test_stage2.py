from __future__ import print_function
import argparse
from ast import arg
import os
import os.path as osp
import re
import random
from utils.geometry import cassini2Equirec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
import cv2
from dataloader import list_deep360_file as lt
from dataloader import MyDeep360Loader as DA
from models import *
from utils import evaluation
import prettytable as pt

parser = argparse.ArgumentParser(description='MODE-Net-stage2')
parser.add_argument('--maxdepth', type=float ,default=1000.0,
                    help='maxium depth')
parser.add_argument('--model', default='UnetRgbConf',
                    help='select model')
parser.add_argument('--dbname', default= None,
                    help='dataset name')
parser.add_argument('--resize', action='store_true', default=False,
                    help='resize the input by downsampling to 1/4 of its original size')
parser.add_argument('--datapath-input', default='/home/jinxueqian/MODE-Net_output_stage1/',
                    help='the datapath of the input')
parser.add_argument('--datapath-gt', default='/data/Share/datasets/Deep360/depth_on_1/',
                    help='the datapath of the groundtruth')
parser.add_argument('--outpath', default='/home/jinxueqian/MODE-Net_results/',
                    help='output path')
parser.add_argument('--batch-size', type=int, default=8,
                    help='batch size')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--projection', default='erp',
                    help='the projection style of the depth map when testing')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dbname == 'Deep360':
    test_depthes, test_confs, test_rgbs, test_gt = lt.listfile_stage2_test(args.datapath_input, args.datapath_gt, False)
elif args.dbname == 'Deep360_soiled':
    test_depthes, test_confs, test_rgbs, test_gt = lt.listfile_stage2_test(args.datapath_input, args.datapath_gt, True)
elif args.dbname == '3D60':
    test_depthes, test_confs, test_rgbs, test_gt = lt.listfile_stage2_test_3d60(args.datapath_input)
else:
    test_depthes, test_confs, test_rgbs, test_gt = lt.listfile_stage2_test_omnifisheye(args.datapath_input, args.datapath_gt)

if args.model == 'MultiviewFusion0':
    model = MultiviewFusion0(args.maxdepth)
elif args.model == 'Baseline':
    model = Baseline(args.maxdepth)
elif args.model == 'Unet':
    model = Unet(args.maxdepth, [32, 64, 128, 256])
elif args.model == 'UnetRgb':
    model = UnetRgb(args.maxdepth, [32, 64, 128, 256])
elif args.model == 'UnetRgbConf':
    if args.dbname == '3D60':
        model = UnetRgbConf(args.maxdepth, [32, 64, 128, 256], {'depth': 4, 'rgb': 6})
    else:
        model = UnetRgbConf(args.maxdepth, [32, 64, 128, 256], {'depth': 12, 'rgb': 12})
elif args.model == 'StereoFusion0':
    model = StereoFusion0()
elif args.model == 'Stereo12':
    model = Stereo12()
elif args.model == 'Stereo13':
    model = Stereo13()
elif args.model == 'Stereo14':
    model = Stereo14()
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(depthes, confs, rgbs, gt):
    model.eval()

    if args.cuda:
        depthes = [depth.cuda() for depth in depthes]
        confs = [conf.cuda() for conf in confs]
        rgbs = [rgb.cuda() for rgb in rgbs]
        gt = gt.cuda()

    with torch.no_grad():
        if args.model == 'Baseline' or args.model == 'Unet':
            output = model(depthes)
        elif args.model == 'UnetRgb':
            output = model(depthes, rgbs)
        else:
            output = model(depthes, confs, rgbs)
        if args.resize:
            output = F.interpolate(output, scale_factor=[2, 2], mode='bicubic', align_corners=True)
        pred = torch.squeeze(output, 1)
    
    if args.projection == 'erp':
        pred = cassini2Equirec(pred.unsqueeze(1))
        gt = cassini2Equirec(gt.unsqueeze(1))

    #---------
    mask = gt <= args.maxdepth
    #----
    eval_metrics = []
    eval_metrics.append(evaluation.mae(pred[mask], gt[mask]))
    eval_metrics.append(evaluation.rmse(pred[mask], gt[mask]))
    eval_metrics.append(evaluation.absrel(pred[mask], gt[mask]))
    eval_metrics.append(evaluation.sqrel(pred[mask], gt[mask]))
    eval_metrics.append(evaluation.silog(pred[mask], gt[mask]))
    eval_metrics.append(evaluation.delta_acc(1, pred[mask], gt[mask]))
    eval_metrics.append(evaluation.delta_acc(2, pred[mask], gt[mask]))
    eval_metrics.append(evaluation.delta_acc(3, pred[mask], gt[mask]))

    return np.array(eval_metrics), pred.data.cpu().numpy(), gt.data.cpu().numpy()

def main():
    TestImgLoader = torch.utils.data.DataLoader(
            DA.myDataLoaderStage2(test_depthes, test_confs, test_rgbs, test_gt, args.resize, False), 
            batch_size= args.batch_size, shuffle= False, num_workers= args.batch_size, drop_last=False)
    #------------- Check the output directories -------
    snapshot_name = osp.splitext(osp.basename(args.loadmodel))[0]
    result_dir = osp.join(args.outpath, args.dbname, snapshot_name)
    depth_pred_path = os.path.join(result_dir, "depth_pred")
    if not os.path.exists(depth_pred_path):
        os.makedirs(depth_pred_path)
    input_png_path = os.path.join(result_dir, "input_png")
    if not os.path.exists(input_png_path):
        os.makedirs(input_png_path)
    gt_png_path = os.path.join(result_dir, "gt_png")
    if not os.path.exists(gt_png_path):
        os.makedirs(gt_png_path)
    rgb_path = os.path.join(result_dir, "rgb")
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path)
    #------------- TESTING -------------------------
    total_eval_metrics = np.zeros(8)
    for batch_idx, (gt_name, depthes, confs, rgbs, gt) in enumerate(TestImgLoader):
        print("\rStage2 测试进度： {:.2f}%".format(100 *  (batch_idx+1)/ len(TestImgLoader)), end='')

        eval_metrics, depth_pred_batch, gt_batch = test(depthes, confs, rgbs, gt)
        total_eval_metrics += eval_metrics

        for i in range(depth_pred_batch.shape[0]):
            name = osp.splitext(osp.basename(gt_name[i]))[0]
            if args.dbname == 'Deep360' or args.dbname == 'Deep360_soiled':
                ep_name = re.findall(r'ep[0-9]_',gt_name[i])[0]
                name = ep_name+name
            # save input png
            for j, depth_input in enumerate(depthes):
                depth_input=np.array(depth_input[i, 0, :, :])
                depth_input=np.log(depth_input-np.min(depth_input)+1)
                depth_input=255*depth_input/np.max(depth_input)
                depth_input=np.clip(depth_input,0,255)
                depth_input = depth_input.astype(np.uint8)
                depth_input = cv2.applyColorMap(depth_input, cv2.COLORMAP_JET)
                cv2.imwrite(input_png_path+'/'+name+"_input_"+str(j+1)+".png", depth_input)
            
            # save gt png
            depth_gt = gt_batch[i]
            depth_gt=np.log(depth_gt-np.min(depth_gt)+1)
            depth_gt=255*depth_gt/np.max(depth_gt)
            depth_gt=np.clip(depth_gt,0,255)
            depth_gt = depth_gt.astype(np.uint8)
            depth_gt = cv2.applyColorMap(depth_gt, cv2.COLORMAP_JET)
            cv2.imwrite(gt_png_path+'/'+name+"_gt.png", depth_gt)


            # save depth pred
            depth_pred = depth_pred_batch[i]
            # np.save(depth_pred_path+'/'+name+"_pred.npy", depth_pred)
            depth_pred=np.log(depth_pred-np.min(depth_pred)+1)
            depth_pred=255*depth_pred/np.max(depth_pred)
            depth_pred=np.clip(depth_pred,0,255)
            depth_pred = depth_pred.astype(np.uint8)
            depth_pred = cv2.applyColorMap(depth_pred, cv2.COLORMAP_JET)
            cv2.imwrite(depth_pred_path+'/'+name+"_pred.png", depth_pred)

    eval_metrics = total_eval_metrics / len(TestImgLoader)
    tb = pt.PrettyTable()
    tb.field_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
    tb.add_row(list(eval_metrics))
    print('\n测试集上的平均表现（%s 投影方式）：\n'%args.projection)
    print(tb)

    # with_rgb 输出不同污损配置的测试结果暂未实现
    # if args.soil and args.soil_detail:
    #     test_12_tmp = test_12[0:int(len(test_12)/3)]
    #     test_13_tmp = test_13[0:int(len(test_12)/3)]
    #     test_14_tmp = test_14[0:int(len(test_12)/3)]
    #     test_23_tmp = test_23[0:int(len(test_12)/3)]
    #     test_24_tmp = test_24[0:int(len(test_12)/3)]
    #     test_34_tmp = test_34[0:int(len(test_12)/3)]
    #     test_depth_tmp = test_depth[0:int(len(test_12)/3)]
    #     TestImgLoader = torch.utils.data.DataLoader(
    #             DA.myDataLoaderStage2(test_12_tmp, test_13_tmp, test_14_tmp, test_23_tmp, test_24_tmp, test_34_tmp, test_depth_tmp, False), 
    #             batch_size= args.batch_size, shuffle= False, num_workers= args.batch_size, drop_last=False)

    #     total_eval_metrics = np.zeros(8)
    #     for batch_idx, (depth12, depth13, depth14, depth23, depth24, depth34, depth) in enumerate(TestImgLoader):
    #         eval_metrics = test(depth12, depth13, depth14, depth23, depth24, depth34, depth)
    #         total_eval_metrics += eval_metrics
    #     eval_metrics = total_eval_metrics / len(TestImgLoader)
    #     tb = pt.PrettyTable()
    #     tb.field_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
    #     tb.add_row(list(eval_metrics))
    #     print('Soiled Type: Glare\n')
    #     print(tb)

    #     test_12_tmp = test_12[int(len(test_12)/3):2*int(len(test_12)/3)]
    #     test_13_tmp = test_13[int(len(test_12)/3):2*int(len(test_12)/3)]
    #     test_14_tmp = test_14[int(len(test_12)/3):2*int(len(test_12)/3)]
    #     test_23_tmp = test_23[int(len(test_12)/3):2*int(len(test_12)/3)]
    #     test_24_tmp = test_24[int(len(test_12)/3):2*int(len(test_12)/3)]
    #     test_34_tmp = test_34[int(len(test_12)/3):2*int(len(test_12)/3)]
    #     test_depth_tmp = test_depth[int(len(test_12)/3):2*int(len(test_12)/3)]
    #     TestImgLoader = torch.utils.data.DataLoader(
    #             DA.myDataLoaderStage2(test_12_tmp, test_13_tmp, test_14_tmp, test_23_tmp, test_24_tmp, test_34_tmp, test_depth_tmp, False), 
    #             batch_size= args.batch_size, shuffle= False, num_workers= args.batch_size, drop_last=False)

    #     total_eval_metrics = np.zeros(8)
    #     for batch_idx, (depth12, depth13, depth14, depth23, depth24, depth34, depth) in enumerate(TestImgLoader):
    #         eval_metrics = test(depth12, depth13, depth14, depth23, depth24, depth34, depth)
    #         total_eval_metrics += eval_metrics
    #     eval_metrics = total_eval_metrics / len(TestImgLoader)
    #     tb = pt.PrettyTable()
    #     tb.field_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
    #     tb.add_row(list(eval_metrics))
    #     print('Soiled Type: Mud\n')
    #     print(tb)

    #     test_12_tmp = test_12[2*int(len(test_12)/3):]
    #     test_13_tmp = test_13[2*int(len(test_12)/3):]
    #     test_14_tmp = test_14[2*int(len(test_12)/3):]
    #     test_23_tmp = test_23[2*int(len(test_12)/3):]
    #     test_24_tmp = test_24[2*int(len(test_12)/3):]
    #     test_34_tmp = test_34[2*int(len(test_12)/3):]
    #     test_depth_tmp = test_depth[2*int(len(test_12)/3):]
    #     TestImgLoader = torch.utils.data.DataLoader(
    #             DA.myDataLoaderStage2(test_12_tmp, test_13_tmp, test_14_tmp, test_23_tmp, test_24_tmp, test_34_tmp, test_depth_tmp, False), 
    #             batch_size= args.batch_size, shuffle= False, num_workers= args.batch_size, drop_last=False)

    #     total_eval_metrics = np.zeros(8)
    #     for batch_idx, (depth12, depth13, depth14, depth23, depth24, depth34, depth) in enumerate(TestImgLoader):
    #         eval_metrics = test(depth12, depth13, depth14, depth23, depth24, depth34, depth)
    #         total_eval_metrics += eval_metrics
    #     eval_metrics = total_eval_metrics / len(TestImgLoader)
    #     tb = pt.PrettyTable()
    #     tb.field_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
    #     tb.add_row(list(eval_metrics))
    #     print('Soiled Type: Water Drop\n')
    #     print(tb)

    #     for soil_cam_num in range(2):
    #         test_12_tmp = []
    #         test_13_tmp = []
    #         test_14_tmp = []
    #         test_23_tmp = []
    #         test_24_tmp = []
    #         test_34_tmp = []
    #         test_depth_tmp = []
    #         for i in range(3):
    #             test_12_tmp += test_12[i*int(len(test_12)/3) + soil_cam_num*int(len(test_12)/6) : i*int(len(test_12)/3) + int(len(test_12)/6) + soil_cam_num*int(len(test_12)/6)]
    #             test_13_tmp += test_13[i*int(len(test_12)/3) + soil_cam_num*int(len(test_12)/6) : i*int(len(test_12)/3) + int(len(test_12)/6) + soil_cam_num*int(len(test_12)/6)]
    #             test_14_tmp += test_14[i*int(len(test_12)/3) + soil_cam_num*int(len(test_12)/6) : i*int(len(test_12)/3) + int(len(test_12)/6) + soil_cam_num*int(len(test_12)/6)]
    #             test_23_tmp += test_23[i*int(len(test_12)/3) + soil_cam_num*int(len(test_12)/6) : i*int(len(test_12)/3) + int(len(test_12)/6) + soil_cam_num*int(len(test_12)/6)]
    #             test_24_tmp += test_24[i*int(len(test_12)/3) + soil_cam_num*int(len(test_12)/6) : i*int(len(test_12)/3) + int(len(test_12)/6) + soil_cam_num*int(len(test_12)/6)]
    #             test_34_tmp += test_34[i*int(len(test_12)/3) + soil_cam_num*int(len(test_12)/6) : i*int(len(test_12)/3) + int(len(test_12)/6) + soil_cam_num*int(len(test_12)/6)]
    #             test_depth_tmp += test_depth[i*int(len(test_12)/3) + soil_cam_num*int(len(test_12)/6) : i*int(len(test_12)/3) + int(len(test_12)/6) + soil_cam_num*int(len(test_12)/6)]
    #         TestImgLoader = torch.utils.data.DataLoader(
    #                 DA.myDataLoaderStage2(test_12_tmp, test_13_tmp, test_14_tmp, test_23_tmp, test_24_tmp, test_34_tmp, test_depth_tmp, False), 
    #                 batch_size= args.batch_size, shuffle= False, num_workers= args.batch_size, drop_last=False)

    #         total_eval_metrics = np.zeros(8)
    #         for batch_idx, (depth12, depth13, depth14, depth23, depth24, depth34, depth) in enumerate(TestImgLoader):
    #             eval_metrics = test(depth12, depth13, depth14, depth23, depth24, depth34, depth)
    #             total_eval_metrics += eval_metrics
    #         eval_metrics = total_eval_metrics / len(TestImgLoader)
    #         tb = pt.PrettyTable()
    #         tb.field_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
    #         tb.add_row(list(eval_metrics))
    #         print('Soiled Camera Number: '+ str(soil_cam_num+1) +'\n')
    #         print(tb)

    #     for spot_num in range(5):
    #         test_12_tmp = []
    #         test_13_tmp = []
    #         test_14_tmp = []
    #         test_23_tmp = []
    #         test_24_tmp = []
    #         test_34_tmp = []
    #         test_depth_tmp = []
    #         for i in range(6):
    #             test_12_tmp += test_12[i*int(len(test_12)/6) + spot_num*int(len(test_12)/30) : i*int(len(test_12)/6) + int(len(test_12)/30) + spot_num*int(len(test_12)/30)]
    #             test_13_tmp += test_13[i*int(len(test_12)/6) + spot_num*int(len(test_12)/30) : i*int(len(test_12)/6) + int(len(test_12)/30) + spot_num*int(len(test_12)/30)]
    #             test_14_tmp += test_14[i*int(len(test_12)/6) + spot_num*int(len(test_12)/30) : i*int(len(test_12)/6) + int(len(test_12)/30) + spot_num*int(len(test_12)/30)]
    #             test_23_tmp += test_23[i*int(len(test_12)/6) + spot_num*int(len(test_12)/30) : i*int(len(test_12)/6) + int(len(test_12)/30) + spot_num*int(len(test_12)/30)]
    #             test_24_tmp += test_24[i*int(len(test_12)/6) + spot_num*int(len(test_12)/30) : i*int(len(test_12)/6) + int(len(test_12)/30) + spot_num*int(len(test_12)/30)]
    #             test_34_tmp += test_34[i*int(len(test_12)/6) + spot_num*int(len(test_12)/30) : i*int(len(test_12)/6) + int(len(test_12)/30) + spot_num*int(len(test_12)/30)]
    #             test_depth_tmp += test_depth[i*int(len(test_12)/6) + spot_num*int(len(test_12)/30) : i*int(len(test_12)/6) + int(len(test_12)/30) + spot_num*int(len(test_12)/30)]
    #         TestImgLoader = torch.utils.data.DataLoader(
    #                 DA.myDataLoaderStage2(test_12_tmp, test_13_tmp, test_14_tmp, test_23_tmp, test_24_tmp, test_34_tmp, test_depth_tmp, False), 
    #                 batch_size= args.batch_size, shuffle= False, num_workers= args.batch_size, drop_last=False)

    #         total_eval_metrics = np.zeros(8)
    #         for batch_idx, (depth12, depth13, depth14, depth23, depth24, depth34, depth) in enumerate(TestImgLoader):
    #             eval_metrics = test(depth12, depth13, depth14, depth23, depth24, depth34, depth)
    #             total_eval_metrics += eval_metrics
    #         eval_metrics = total_eval_metrics / len(TestImgLoader)
    #         tb = pt.PrettyTable()
    #         tb.field_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
    #         tb.add_row(list(eval_metrics))
    #         print('Spot Number: '+ str(spot_num+1) +'\n')
    #         print(tb)

    #     for soil_rate in range(4):
    #         test_12_tmp = []
    #         test_13_tmp = []
    #         test_14_tmp = []
    #         test_23_tmp = []
    #         test_24_tmp = []
    #         test_34_tmp = []
    #         test_depth_tmp = []
    #         for i in range(30):
    #             test_12_tmp += test_12[i*int(len(test_12)/30) + soil_rate*int(len(test_12)/120) : i*int(len(test_12)/30) + int(len(test_12)/120) + soil_rate*int(len(test_12)/120)]
    #             test_13_tmp += test_13[i*int(len(test_12)/30) + soil_rate*int(len(test_12)/120) : i*int(len(test_12)/30) + int(len(test_12)/120) + soil_rate*int(len(test_12)/120)]
    #             test_14_tmp += test_14[i*int(len(test_12)/30) + soil_rate*int(len(test_12)/120) : i*int(len(test_12)/30) + int(len(test_12)/120) + soil_rate*int(len(test_12)/120)]
    #             test_23_tmp += test_23[i*int(len(test_12)/30) + soil_rate*int(len(test_12)/120) : i*int(len(test_12)/30) + int(len(test_12)/120) + soil_rate*int(len(test_12)/120)]
    #             test_24_tmp += test_24[i*int(len(test_12)/30) + soil_rate*int(len(test_12)/120) : i*int(len(test_12)/30) + int(len(test_12)/120) + soil_rate*int(len(test_12)/120)]
    #             test_34_tmp += test_34[i*int(len(test_12)/30) + soil_rate*int(len(test_12)/120) : i*int(len(test_12)/30) + int(len(test_12)/120) + soil_rate*int(len(test_12)/120)]
    #             test_depth_tmp += test_depth[i*int(len(test_12)/30) + soil_rate*int(len(test_12)/120) : i*int(len(test_12)/30) + int(len(test_12)/120) + soil_rate*int(len(test_12)/120)]
    #         TestImgLoader = torch.utils.data.DataLoader(
    #                 DA.myDataLoaderStage2(test_12_tmp, test_13_tmp, test_14_tmp, test_23_tmp, test_24_tmp, test_34_tmp, test_depth_tmp, False), 
    #                 batch_size= args.batch_size, shuffle= False, num_workers= args.batch_size, drop_last=False)

    #         total_eval_metrics = np.zeros(8)
    #         for batch_idx, (depth12, depth13, depth14, depth23, depth24, depth34, depth) in enumerate(TestImgLoader):
    #             eval_metrics = test(depth12, depth13, depth14, depth23, depth24, depth34, depth)
    #             total_eval_metrics += eval_metrics
    #         eval_metrics = total_eval_metrics / len(TestImgLoader)
    #         tb = pt.PrettyTable()
    #         tb.field_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
    #         tb.add_row(list(eval_metrics))
    #         print('Soil Rate: '+ str((soil_rate+1)*5) +'%\n')
    #         print(tb)


if __name__ == '__main__':
   main()