#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jaschn/DeepLearningProject/blob/main/InstanceColorization/Run_training_InstAware.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## Environment Settings

# In[1]:

import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

from os.path import join, isfile, isdir
from os import listdir
import os
from argparse import ArgumentParser

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from IPython import get_ipython
import torch
from tqdm import tqdm


import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import trange, tqdm

from fusion_dataset import *
from util import util
import os

opt = type('', (), {})()
opt.train_img_dir = './train_sample/train2017'
opt.fineSize = 256
opt.batch_size = 5
opt.loadSize = 256
opt.input_nc = 1
opt.output_nc = 2
opt.ngf = 64
opt.ndf = 64
opt.which_model_netD = 'basic'
opt.which_model_netG = 'siggraph'
opt.n_layers_D = 3
opt.gpu_ids = [0]
opt.dataset_mode = 'aligned'
opt.which_direction = 'AtoB'
opt.nThreads = 4
opt.checkpoints_dir = './checkpoints'
opt.norm = 'batch'
opt.serial_batches = 'store_true'
opt.display_winsize = 256
opt.display_id = 1
opt.display_server = 'http://localhost'
opt.display_port = 8097
opt.no_dropout = False
opt.checkpoints_dir = './checkpoints'
opt.max_dataset_size = float("inf")
opt.resize_or_crop = 'resize_and_crop'
opt.no_flip = False
opt.init_type = 'normal'
opt.verbose = False
opt.suffix = ''
opt.ab_norm = 110
opt.ab_max = 110
opt.ab_quant = 10
opt.l_norm = 100
opt.l_cent = 50
opt.mask_cent = 5
opt.suffix = ''
opt.sample_p = 1.0
opt.sample_Ps = '+'
opt.results_dir = './results/'
opt.classification = False
opt.phase = 'val'
opt.which_epoch = 'latest'
opt.how_many = 200
opt.aspect_ratio = 1.0
opt.load_model = True
opt.half = False
opt.stage = 'fusion'
opt.model = 'train'
opt.name = 'coco_mask'
opt.display_freq = 500
opt.display_ncols = 3
opt.update_html_freq = 10000
opt.print_freq = 500
opt.save_latest_freq = 5000
opt.save_epoch_freq = 1
opt.epoch_count = 0
opt.niter = 1
opt.niter_decay = 1
opt.beta1 = 0.9
opt.lr = 0.00005
opt.no_lsgan = False
opt.lambda_GAN = 0.
opt.lambda_A = 1.
opt.lambda_B = 1.
opt.lambda_identity = 0.5

opt.pool_size = 50
opt.no_html = False
opt.lr_policy = 'lambda'
opt.lr_decay_iters = 50
opt.avg_loss_alpha = .986
opt.isTrain = True


opt.A = 2 * opt.ab_max / opt.ab_quant + 1
opt.B = opt.A

dataset = Training_Fusion_Dataset(opt, 1)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)

dataset_size = len(dataset)
print('#training images = %d' % dataset_size)

model = create_model(opt)
model.setup(opt)

opt.display_port = 8098
# visualizer = Visualizer(opt)
total_steps = 0

if opt.stage == 'full' or opt.stage == 'instance':
  for epoch in trange(opt.epoch_count, opt.niter + opt.niter_decay, desc='epoch', dynamic_ncols=True):
    epoch_iter = 0

    for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        data_raw['rgb_img'] = [data_raw['rgb_img']]
        data_raw['gray_img'] = [data_raw['gray_img']]

        input_data = util.get_colorization_data(data_raw['gray_img'], opt, p=1.0, ab_thresh=0)
        gt_data = util.get_colorization_data(data_raw['rgb_img'], opt, p=1.0, ab_thresh=10.0)
        if gt_data is None:
            continue
        if(gt_data['B'].shape[0] < opt.batch_size):
            continue
        input_data['B'] = gt_data['B']
        input_data['hint_B'] = gt_data['hint_B']
        input_data['mask_B'] = gt_data['mask_B']

        visualizer.reset()
        model.set_input(input_data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            losses = model.get_current_losses()
            if opt.display_id > 0:
              ("")
                # visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

    if epoch % opt.save_epoch_freq == 0:
        model.save_networks('latest')
        model.save_networks(epoch)
    model.update_learning_rate()
elif opt.stage == 'fusion':
  for epoch in trange(0, 1 + 1, desc='epoch', dynamic_ncols=True):
    epoch_iter = 0

    for data_raw in tqdm(dataset_loader, desc='batch', dynamic_ncols=True, leave=False):
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        box_info = data_raw['box_info'][0]
        box_info_2x = data_raw['box_info_2x'][0]
        box_info_4x = data_raw['box_info_4x'][0]
        box_info_8x = data_raw['box_info_8x'][0]
        cropped_input_data = util.get_colorization_data(data_raw['cropped_gray'], opt, p=1.0, ab_thresh=0)
        cropped_gt_data = util.get_colorization_data(data_raw['cropped_rgb'], opt, p=1.0, ab_thresh=10.0)
        full_input_data = util.get_colorization_data(data_raw['full_gray'], opt, p=1.0, ab_thresh=0)
        full_gt_data = util.get_colorization_data(data_raw['full_rgb'], opt, p=1.0, ab_thresh=10.0)
        if cropped_gt_data is None or full_gt_data is None:
            continue
        cropped_input_data['B'] = cropped_gt_data['B']
        full_input_data['B'] = full_gt_data['B']
        # visualizer.reset()
        model.set_input(cropped_input_data)
        model.set_fusion_input(full_input_data, [box_info, box_info_2x, box_info_4x, box_info_8x])
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            losses = model.get_current_losses()
            if opt.display_id > 0:
              ("")
                # visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
    if epoch % opt.save_epoch_freq == 0:
        model.save_fusion_epoch(epoch)
    model.update_learning_rate()
else:
  print('Error! Wrong stage selection!')
  exit()


# In[ ]:


# save coco_mask
get_ipython().system('zip -r coco_mask.zip checkpoints/coco_mask')
from google.colab import files
# files.download("coco_mask.zip")
import shutil
shutil.move('coco_mask.zip', '/content/drive/MyDrive/')


