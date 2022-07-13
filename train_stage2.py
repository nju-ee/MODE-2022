from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import list_deep360_file as lt
from dataloader import MyDeep360Loader as DA
from models import *
from utils import evaluation
import prettytable as pt
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='MODE-Fusion')
parser.add_argument('--maxdepth', type=float ,default=1000.0,
                    help='maximum depth in meters')
parser.add_argument('--model', default='UnetRgbConf',
                    help='select model')
parser.add_argument('--dbname', default= "Deep360",
                    help='dataset name')
parser.add_argument('--soil', action='store_true', default=False,
                    help='train fusion network from soiled data (only for Deep360)')
parser.add_argument('--resize', action='store_true', default=False,
                    help='resize the input by downsampling to 1/2 of its original size')
parser.add_argument('--datapath-input', default='./MODE-Disparity_output/',
                    help='the path of the input of stage2, which is just the output of stage1')
parser.add_argument('--datapath-dataset', default='./datasets/Deep360/',
                    help='the path of the dataset')
parser.add_argument('--epochs', type=int, default=150,
                    help='the number of epochs for training')
parser.add_argument('--epoch-start', type=int, default=0,
                    help='change this if the training was broken and you want to continue from the breakpoint')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--loadmodel', default= None,
                    help='load model path')
parser.add_argument('--savemodel', default='./checkpoint/stage2/',
                    help='save model path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dbname == 'Deep360':
    train_depthes, train_confs, train_rgbs, train_gt, val_depthes, val_confs, val_rgbs, val_gt = lt.listfile_stage2_train(args.datapath_input, args.datapath_dataset, args.soil)
elif args.dbname == '3D60':
    train_depthes, train_confs, train_rgbs, train_gt, val_depthes, val_confs, val_rgbs, val_gt = lt.listfile_stage2_train_3d60(args.datapath_input)
else:
    train_depthes, train_confs, train_rgbs, train_gt, val_depthes, val_confs, val_rgbs, val_gt = lt.listfile_stage2_train_omnifisheye(args.datapath_input, args.datapath_dataset)

TrainImgLoader = torch.utils.data.DataLoader(
        DA.myDataLoaderStage2(train_depthes, train_confs, train_rgbs, train_gt, resize=args.resize, training=True), 
        batch_size= args.batch_size, shuffle= True, num_workers = args.batch_size, drop_last=False)

ValImgLoader = torch.utils.data.DataLoader(
        DA.myDataLoaderStage2(val_depthes, val_confs, val_rgbs, val_gt, resize=False, training=False), 
        batch_size= 8, shuffle= False, num_workers= 8, drop_last=False)

if args.model == 'MultiviewFusion0':
    model = MultiviewFusion0(args.maxdepth)
elif args.model == 'Baseline':
    model = Baseline(args.maxdepth)
elif args.model == 'MultiviewFusion2':
    model = MultiviewFusion2(args.maxdepth)
elif args.model == 'Unet':
    model = Unet(args.maxdepth, [32, 64, 128, 256])
elif args.model == 'UnetRgb':
    model = UnetRgb(args.maxdepth, [32, 64, 128, 256])
elif args.model == 'UnetRgbConf':
    if args.dbname == '3D60':
        model = UnetRgbConf(args.maxdepth, [32, 64, 128, 256], {'depth': 4, 'rgb': 6})
    else:
        model = UnetRgbConf(args.maxdepth, [32, 64, 128, 256], {'depth': 12, 'rgb': 12})
        # model = UnetRgbConf(args.maxdepth, [32, 64, 128, 256], {'depth': 2, 'rgb': 3})
elif args.model == 'StereoFusion0':
    model = StereoFusion0()
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

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
# optimizer = optim.SGD(model.parameters(), lr = 0.005, momentum=0.9)

def silog_loss(lamda, pred, gt):
    mask1 = gt > 0
    mask2 = pred > 0
    mask = mask1 * mask2
    d = torch.log(pred[mask])-torch.log(gt[mask])
    return torch.mean(torch.square(d)) - lamda * torch.square(torch.mean(d))

def log_loss_with_soil_mask(pred, gt, soil_mask, lamda_soil):
    mask1 = gt > 0
    mask2 = pred > 0
    mask = mask1 * mask2
    d = torch.log(pred[mask])-torch.log(gt[mask])
    d = torch.square(d)
    loss_soil = torch.mean(d[soil_mask[mask]>0])
    loss_clean = torch.mean(d[soil_mask[mask]==0])
    return lamda_soil * loss_soil + (1-lamda_soil) * loss_clean

def train(depthes, confs, rgbs, gt):
    model.train()

    if args.cuda:
        depthes = [depth.cuda() for depth in depthes]
        confs = [conf.cuda() for conf in confs]
        rgbs = [rgb.cuda() for rgb in rgbs]
        gt = gt.cuda()

    #---------
    mask = gt < args.maxdepth
    mask.detach_()
    #----

    optimizer.zero_grad()
    
    if args.model == 'Baseline' or args.model == 'Unet':
        output = model(depthes)
    elif args.model == 'UnetRgb':
        output = model(depthes, rgbs)
    else:
        output = model(depthes, confs, rgbs)
    output = torch.squeeze(output, 1)
    # loss = F.smooth_l1_loss(output, gt, reduction='mean')
    loss = silog_loss(0.5, output[mask], gt[mask])

    loss.backward()
    optimizer.step()

    return loss.data

def val(depthes, confs, rgbs, gt):
    model.eval()

    if args.cuda:
        depthes = [depth.cuda() for depth in depthes]
        confs = [conf.cuda() for conf in confs]
        rgbs = [rgb.cuda() for rgb in rgbs]
        gt = gt.cuda()

    #---------
    mask = gt < args.maxdepth
    #----

    with torch.no_grad():
        if args.model == 'Baseline' or args.model == 'Unet':
            output = model(depthes)
        elif args.model == 'UnetRgb':
            output = model(depthes, rgbs)
        else:
            output = model(depthes, confs, rgbs)
        pred = torch.squeeze(output, 1)

    eval_metrics = []
    eval_metrics.append(evaluation.mae(pred[mask], gt[mask]))
    eval_metrics.append(evaluation.rmse(pred[mask], gt[mask]))
    eval_metrics.append(evaluation.absrel(pred[mask], gt[mask]))
    eval_metrics.append(evaluation.sqrel(pred[mask], gt[mask]))
    eval_metrics.append(evaluation.silog(pred[mask], gt[mask]))
    eval_metrics.append(evaluation.delta_acc(1, pred[mask], gt[mask]))
    eval_metrics.append(evaluation.delta_acc(2, pred[mask], gt[mask]))
    eval_metrics.append(evaluation.delta_acc(3, pred[mask], gt[mask]))

    return np.array(eval_metrics)

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    # if epoch<5:
    #     lr = 0.001
    # elif epoch<10:
    #     lr = 0.0001
    # elif epoch<30:
    #     lr = 0.00001
    # else:
    #     lr = 0.000001
    # print('learning rate = ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    if not os.path.exists(args.savemodel+'log/'):
        os.makedirs(args.savemodel+'log/')
    writer = SummaryWriter(args.savemodel+'log/', purge_step = args.epoch_start)

    start_full_time = time.time()
    for epoch in range(0, args.epochs):
        # print('This is %d-th epoch' %(epoch+args.epoch_start))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch+args.epoch_start)

        #--- TRAINING ---#
        for batch_idx, (_, depthes, confs, rgbs, gt) in enumerate(TrainImgLoader):
            loss = train(depthes, confs, rgbs, gt)
            print("\rStage2 Epoch"+str(epoch+args.epoch_start)+" 训练进度： {:.2f}%".format(100 *  (batch_idx+1)/ len(TrainImgLoader)), end='')
            total_train_loss += loss
        writer.add_scalar('Training Loss', total_train_loss/len(TrainImgLoader), epoch+args.epoch_start)

        #--- SAVING ---#
        savefilename = args.savemodel+'/checkpoint_stage2_'+args.model+'_lr_'+str(args.lr)+'_epoch'+str(epoch+args.epoch_start)+'.tar'
        torch.save({
                'state_dict': model.state_dict()
            }, savefilename)
        
        #--- VALIDATION ---#
        total_eval_metrics = np.zeros(8)
        for batch_idx, (_, depthes, confs, rgbs, gt) in enumerate(ValImgLoader):
            print("\rStage2 Epoch"+str(epoch+args.epoch_start)+" 验证进度： {:.2f}%".format(100 *  (batch_idx+1)/ len(ValImgLoader)), end='')
            eval_metrics = val(depthes, confs, rgbs, gt)
            total_eval_metrics += eval_metrics

        eval_metrics = total_eval_metrics / len(ValImgLoader)
        # tb = pt.PrettyTable()
        # tb.field_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
        # tb.add_row(list(eval_metrics))
        # print('\n')
        # print(tb)
        writer.add_scalar('MAE', eval_metrics[0], epoch+args.epoch_start)
        writer.add_scalar('RMSE', eval_metrics[1], epoch+args.epoch_start)
        writer.add_scalar('AbsRel', eval_metrics[2], epoch+args.epoch_start)
        writer.add_scalar('SqRel', eval_metrics[3], epoch+args.epoch_start)
        writer.add_scalar('SILog', eval_metrics[4], epoch+args.epoch_start)
        writer.add_scalar('δ1', eval_metrics[5], epoch+args.epoch_start)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

    writer.close()


if __name__ == '__main__':
   main()