import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob

import argparse

def torchToCv(input):
    input = input.squeeze(0)
    inputCv = input.numpy()
    inputCv = np.transpose(inputCv, (1,2,0))
    return inputCv

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MNAD")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
    parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
    parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='./dataset/', help='directory of data')
    parser.add_argument('--model_dir', type=str, help='directory of model')
    parser.add_argument('--m_items_dir', type=str, help='directory of model')
    parser.add_argument('--m_dis_dir', type=str, help='directory of model')

    args = parser.parse_args()
    #my code: load model
    exp_path = './exp/'
    model_dir = 'model.pth'
    m_items_dir = 'keys.pt'
    m_dis_dir = 'dis.pt'
    # args.model_dir = 'Ped2_prediction_model.pth'
    # args.m_items_dir = 'Ped2_prediction_keys.pt'
    args.model_dir = exp_path + args.dataset_type + '/log_position/' + model_dir
    args.m_items_dir = exp_path + args.dataset_type + '/log_position/' + m_items_dir
    args.m_dis_dir = exp_path + args.dataset_type + '/log_position/' + m_dis_dir

    torch.manual_seed(2020)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if args.gpus is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    else:
        gpus = ""
        for i in range(len(args.gpus)):
            gpus = gpus + args.gpus[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    test_folder = args.dataset_path+args.dataset_type+"/testing/frames"

    # Loading dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
                 transforms.ToTensor(),
                 ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_size = len(test_dataset)

    test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    model = torch.load(args.model_dir)
    model.cuda()
    m_items = torch.load(args.m_items_dir)
    m_dis = torch.load(args.m_dis_dir)

    labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')
    if args.dataset_type == 'shanghai':
        labels = np.expand_dims(labels, 0)

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('\\')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('\\')[-1]
        labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('\\')[-1]]['length']
    m_items_test = m_items.clone()
    m_dis_test = m_dis.clone()
    timing_signal = get_timing_signal_1d(args.h, args.w)
    timing_signal = timing_signal.repeat(args.t_length - 1, 1, 1)

    model.eval()

    # my code: make a circular in img
    f = True
    make_circular = False
    get_memory_dis = False
    circularPath = r'dataset\ped2\mytesting\frames\circular_black.jpg'
    circul_H = 200
    circul_W = 200
    test = -1
    w = 1
    x = 0
    y = 20
    step = 1
    circular = cv2.imread(circularPath)
    circular = cv2.resize(circular,(circul_W,circul_H),interpolation=cv2.INTER_AREA)
    circularGray = cv2.cvtColor(circular, cv2.COLOR_BGR2GRAY)
    circular = cv2.bitwise_not(circular)
    ret, circular = cv2.threshold(circular, 100, 255, cv2.THRESH_TRUNC)
    circular = circular / 255
    circular = circular.astype('float32')
    ret, mask = cv2.threshold(circularGray, 150, 255, cv2.THRESH_BINARY)


    for k,(imgs) in enumerate(test_batch):

        if k == label_length-4*(video_num+1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('\\')[-1]]['length']

        # my code: make a circular in img
        if make_circular:
            with torch.no_grad():
                for i in range(5):
                    img = (imgs[:, i*3 : i*3+3]+1)/2
                    imgs_cv = torchToCv(img)
                    if w >= circul_W:
                        roi = imgs_cv[y: y+circul_H, x: x+circul_W]
                        img_bg = cv2.bitwise_and(roi, roi, mask=mask)
                        dis = cv2.add(img_bg,circular)
                        imgs_cv[y: y + circul_H, x: x + circul_W] = dis
                        imgs[:, i*3 : i*3+3] = torch.from_numpy(imgs_cv).permute(2,0,1).unsqueeze(0)*2-1
                        if x+circul_W < 256:
                            x = x + step
                        else:
                            x = 0
                    else:
                        roi = imgs_cv[y : y+circul_H, 0 : w]
                        img_bg = cv2.bitwise_and(roi, roi, mask=mask[:, circul_W-w : circul_W])
                        dis = cv2.add(img_bg, circular[:, circul_W-w : circul_W])
                        imgs_cv[y: y + circul_H, 0: w] = dis
                        imgs[:, i * 3: i * 3 + 3] = torch.from_numpy(imgs_cv).permute(2, 0, 1).unsqueeze(0) * 2 - 1
                        w = w + step

        img = (imgs[:, 12:] + 1) / 2

        # imgs[:, 0:12] = imgs[:, 0:12] + timing_signal #timing_signal 空间编码
        imgs = Variable(imgs).cuda()

        outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss, m_dis_test = model.forward(imgs[:,0:3*4], m_items_test, m_dis_test, False, False)
        mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0,3*4:]+1)/2)).item()
        mse_feas = compactness_loss.item()

        # my code: show result
        with torch.no_grad():
            imgCV = torchToCv(imgs[:,12:15].cpu())
            outputsCV = torchToCv(outputs.cpu())
            irec = cv2.absdiff(imgCV,outputsCV)
            cv2.imshow('true', torchToCv(img))
            cv2.imshow('input', (imgCV+1)/2)
            cv2.imshow('output', (outputsCV + 1) / 2)
            cv2.imshow('rec', irec)
            if f:
                cv2.waitKey(0)
                f = False
            else:
                cv2.waitKey(300)

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:,3*4:])

        if point_sc < args.th:
            query = F.normalize(feas, dim=1)
            query = query.permute(0,2,3,1) # b X h X w X d
            m_items_test, m_dis_test = model.memory.update(query, m_items_test, m_dis_test, False, True)

        psnr_list[videos_list[video_num].split('\\')[-1]].append(psnr(mse_imgs))
        feature_distance_list[videos_list[video_num].split('\\')[-1]].append(mse_feas)


    # Measuring the abnormality score and the AUC
    anomaly_score_total_list = []
    for video in sorted(videos_list):
        video_name = video.split('\\')[-1]
        anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]),
                                         anomaly_score_list_inv(feature_distance_list[video_name]), args.alpha)

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

    print('The result of ', args.dataset_type)
    print('AUC: ', accuracy*100, '%')
