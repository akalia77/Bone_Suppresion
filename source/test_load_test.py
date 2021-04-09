# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:26:14 2021

@author: dhkim
"""

import torch
import cv2
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
from pydicom import dcmread

from PIL import Image, ImageFont, ImageDraw

from datasets import load_dataset
from noise2noise import Noise2Noise

from argparse import ArgumentParser


import torchvision.transforms.functional as tvF

import torchvision
import timeit
from torch.utils import mkldnn

#%% read dicom file 

def readDicomfile(path):
    
        
    
    ds = dcmread(path )
    
    dimImg = ds.pixel_array
    
    # wfname = f"{file_Name}_{ds.Columns}x{ds.Rows}.raw"
    # npImg.tofile(wfname )       
    
    # print(f"writefile:{wfname}")
   
    return dimImg
    


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
        choices=['gaussian', 'grid','poisson', 'text', 'mc'], default='gaussian', type=str)
    parser.add_argument('-v', '--noise-param', help='noise parameter (e.g. sigma for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=256, type=int)

    args = parser.parse_args(
                                [
                                # "--data","../data/test",
                                "--data","../data/test_grid",
                                # "--load-ckpt","../ckpts/gaussian/n2n-gaussian.pt",
                                # "--noise-type","gaussian",
                                # "--load-ckpt","../ckpts/text-1643/n2n-epoch9-0.04232.pt",
                                "--noise-type","grid",
                                "--load-ckpt","../saved/bone-clean-210331-1723/n2n-epoch999-1.01681.pt",
                                
                                
                                
                                # "--noise-type","text",
                                # "--noise-param","50",
                                "--noise-param","0.4",
                                # "--crop-size","128",
                                "--crop-size","128",
                                "--show-output","3",
                                "--cuda",
                                ]
                            )

    # return parser.parse_args()
    return args




"""Tests Noise2Noise."""

# loadFileName = "Before_grid_suppression_chest_phantom_3072x3072.raw"
# loadFileName = "./testImgs/BonTech_4_Skull-phantom_110Lp-grid_3072x3072.raw"
# loadFileName = "./testImgs/BonTech_3_Chest-phantom_110Lp-grid_3072x3072.raw"
loadFileName = "../TestImg/grid012.dcm"

img_w=3072
img_h=3072
# Parse test parameters
params = parse_args()

# Initialize model and test
n2n = Noise2Noise(params, trainable=False)
params.redux = False
params.clean_targets = True

n2n.load_model(params.load_ckpt)
# d_img=n2n.testOnefile(loadFileName,img_w, img_h)

# imgArray = np.fromfile(loadFileName, dtype=np.uint16).reshape(img_w,img_h)

# inMax = imgArray.max()
devicetype = "cuda"
# devicetype = "cpu"

diImg= readDicomfile(loadFileName)

outImg= n2n.bonePredictOut(diImg)



timeS = timeit.default_timer()
# d_imgO=n2n.testOneFileCrop(imgArray,img_w, img_h,256,16,devicetype )

timeE =timeit.default_timer()
print("run:: %f sec "%(timeE-timeS))

# d_img= (d_imgO/1000)* inMax

# d_img=np.array(d_img).astype(np.uint16)


# d_img.tofile("./testOutImage/test0315_gpu_BonTech_3_Chest-phantom_110Lp-grid_3072x3072.raw")
# d_img.tofile("test0217_newImg_BonTech_3_Chest-phantom_110Lp-grid_pad16_256_3072x3072.raw")

# d_img

# imgnp=d_img.squeeze().cpu().numpy()

# o_=tvF.to_pil_image( d_img.squeeze())


#%%save the script moduel 

scSaveName = "n2n_torch_script_grid_grayscale_t0.pt"

loadFileName = "testImg_jpg_grid_480_x640_uint16.raw"
r_w=480
r_h=640

imgArray = np.fromfile(loadFileName, dtype=np.uint16).reshape(r_w,r_h)

# imgArray = imgArray/imgArray.max()
img =Image.fromarray(imgArray)

source = tvF.to_tensor(img)

source = torch.unsqueeze(source,0).cuda()


torch_script_module_sm = torch.jit.script(n2n)

torch_script_module = torch.jit.trace(n2n,source)
torch_script_module.save(scSaveName )



