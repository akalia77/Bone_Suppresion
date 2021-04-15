#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datasets import load_dataset
from noise2noise import Noise2Noise

from argparse import ArgumentParser


import torchvision.transforms.functional as tvF

#%%

def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='../data')
    parser.add_argument('--load-ckpt', help='load model checkpoint')
    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'grid','poisson', 'text', 'mc','bone'], default='gaussian', type=str)
    parser.add_argument('-v', '--noise-param', help='noise parameter (e.g. sigma for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=256, type=int)

    args = parser.parse_args(
                                [
                                # "--data","../data/test",
                                "--data","../datasets/valid_bone",
                                
                                # "--load-ckpt","../ckpts/gaussian/n2n-gaussian.pt",
                                # "--noise-type","gaussian",
                                # "--load-ckpt","../ckpts/text-1643/n2n-epoch9-0.04232.pt",
                                "--noise-type","bone",
                                # "--load-ckpt","../ckpts/grid-1759/n2n-epoch120-0.00099.pt",
                                # "--load-ckpt","../ckpts/grid-1058/n2n-epoch18-0.00068.pt",
                                # "--load-ckpt","../saved/grid_0123/n2n-epoch446-0.00522.pt",
                                # "--load-ckpt","../saved/grid-clean-210217/n2n-epoch993-5.26789.pt",
                                "--load-ckpt","../saved/bone-clean-210409-1653-l2/n2n-epoch999-68.65568.pt",
                                
                                # "--noise-type","text",
                                # "--noise-param","50",
                                "--noise-param","0.4",
                                # "--crop-size","128",
                                "--crop-size","1024",
                                "--show-output","3",
                                "--cuda",
                                ]
                            )

    # return parser.parse_args()
    return args


if __name__ == '__main__':
    """Tests Noise2Noise."""

    # Parse test parameters
    params = parse_args()

    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False)
    params.redux = False
    params.clean_targets = True
    test_loader = load_dataset(params.data, 0, params, shuffled=False, single=True)
    n2n.load_model(params.load_ckpt)
    
    n2n.test(test_loader, show=params.show_output)


#%% test load and run 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
b=Image.open("noise_source.jpg")


plt.imshow(inImg)
sm_load=torch.jit.load("n2n_torch_script_t0.pt")

sm_load.state_dict()

inImg = cv2.imread("noise_source.jpg")
#c=tvF.to_pil_image(b)
a=cv2.imread("noise_source.jpg")
out_t=sm_load(inTensor_gpu)

#%% load tensor from image 

sm_load=torch.jit.load("n2n_torch_script_t0.pt")

# b=Image.open("test_src_0.jpg")
b=Image.open("noise_source.jpg").convert('RGB')

# inTensor=torch.unsqueeze(tvF.to_tensor(b),0).cuda()
inTensor=torch.unsqueeze(tvF.to_tensor(tmp),0).cuda()


out_i= torch.squeeze(sm_load(inTensor))

o_img = tvF.to_pil_image(out_i)



#%% test 


load_fileName = "n2n_torch_script_text_t0.pt"
# load_fileName = "n2n_torch_jit_script.pt"
sm_load=torch.jit.load(load_fileName)

sm_load.eval()

# img_path_test = "test_src_2_full.jpg"
img_path_test = "test_src_text_0.jpg"
img =  Image.open(img_path_test).convert('RGB')
# img =  Image.open(img_path).convert('RGB')

# tmp = self._corrupt(img)

inTensor=torch.unsqueeze(tvF.to_tensor(img),0).cuda()

out_i= torch.squeeze(sm_load(inTensor))

o_img = tvF.to_pil_image(out_i)

o_img.show()


#%% full mode test
from PIL import Image
# model = Noise2Noise(params, trainable=False)
model = torch.load("n2n_entire_save.pt")
model.eval()


img_path_test = "test_src_2_full.jpg"
img =  Image.open(img_path_test).convert('BGR')

inTensor=torch.unsqueeze(tvF.to_tensor(img),0).cuda()

out_i= torch.squeeze(model(inTensor))

o_img = tvF.to_pil_image(out_i)

o_img.show()


#%%
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])
plt.ylabel('y-label')
plt.show()

# %%
