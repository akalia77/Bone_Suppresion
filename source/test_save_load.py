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

from PIL import Image, ImageFont, ImageDraw

from datasets import load_dataset
from noise2noise import Noise2Noise

from argparse import ArgumentParser


import torchvision.transforms.functional as tvF

import torchvision
import timeit
from torch.utils import mkldnn


#%% test 
# scM = torch.jit.load("n2n_gridClean_0223_saveModel_script.pt")
# scM = torch.jit.load("../saved/grid-clean-210312-cpu_test/n2n_torch_script_grid_grayscale_t0_cpu.pt")


scM = torch.jit.load("./n2n_torch_script_grid_grayscale_t0.pt")



loadFileName = "./testOutImage/test_0223_source_256x256.raw"
# loadFileName = "./testImgs/BonTech_3_Chest-phantom_110Lp-grid_3072x3072.raw"

# imgArray = np.fromfile(loadFileName, dtype=np.float32).reshape(256,256)

img_w=256
img_h=256

imgArray = np.fromfile(loadFileName, dtype=np.float32).reshape(img_w,img_h)

cpu_is = True

if cpu_is== True :
    scM = scM.cpu()

inMax = imgArray.max()
source = tvF.to_tensor(imgArray)
# sourceIn = torch.unsqueeze(source,0).cuda()
sourceIn = torch.unsqueeze(source,0)


ts= timeit.default_timer()
# denoised_img = self.model(sourceIn).detach()
out_=scM(sourceIn)
te= timeit.default_timer()
print("m_run:: %f sec "%(te-ts))

ts= timeit.default_timer()
# denoised_img = self.model(sourceIn).detach()
out_=scM(sourceIn)
te= timeit.default_timer()
print("m_run:: %f sec "%(te-ts))

ts= timeit.default_timer()
# denoised_img = self.model(sourceIn).detach()
out_=scM(sourceIn)
te= timeit.default_timer()
print("m_run:: %f sec "%(te-ts))


ts= timeit.default_timer()
# denoised_img = self.model(sourceIn).detach()
out_=scM(sourceIn)
te= timeit.default_timer()
print("m_run:: %f sec "%(te-ts))


out = out_.detach().squeeze().cpu()

plt.imshow(out)




#%% test load and run 


scM = torch.jit.load("n2n_gridClear_0223_2_saveModel_script.pt")

# loadFileName = "./test_0223_source_256x256.raw"
loadFileName = "./testImgs/BonTech_3_Chest-phantom_110Lp-grid_3072x3072.raw"

# imgArray = np.fromfile(loadFileName, dtype=np.float32).reshape(256,256)

img_w=3072
img_h=3072

imgArray = np.fromfile(loadFileName, dtype=np.uint16).reshape(img_w,img_h)

inMax = imgArray.max()
# source = tvF.to_tensor(imgArray)
# sourceIn = torch.unsqueeze(source,0).cuda()

# out_=scM(sourceIn)

# out = out_.detach().squeeze().cpu()

# plt.imshow(out)

w=img_w
h=img_h

c_size = 256 
b_size = 16

scM.train(False)
        
# loadFileName = "testImg_jpg_grid_480_x640_uint16.raw"
# loadFileName = inPath

# imgArray = np.fromfile(loadFileName, dtype=np.uint16).reshape(w,h)

imgArray = imgArray/imgArray.max()
img =Image.fromarray(imgArray)

img = np.array(img).astype(np.float32)

img = img/img.max()*1000 # org 

c_w=c_size
c_h=c_size

count_w = w/c_w
count_h = h/c_h

dout_imgs=[]
hImg=[]
vImg=[]




for idy in range(int(count_h)):
    tempimg=[]
    
    if len(hImg) !=0 :
        vImg.append(hImg )
    for idx in range(int(count_w)):                        
        sx=idx*c_size
        sy=idy*c_size
            
        cImg = img[sy:(idy+1)*c_h,sx:(idx+1)*c_w]
        
        pad = b_size*2
        pImg=np.pad(cImg ,(pad ,pad ),'edge')
        
        # plt.imshow(cImg)
        # plt.pause(1)               
        
        source = tvF.to_tensor(pImg)
        
        
        
        
        # sourcePad=torch.zeros(1,c_size+pad *2,c_size+pad *2)
        
        # sourcePad[:,pad:-pad,pad:-pad]=sourceIn
        
        sourceIn = torch.unsqueeze(source,0).cuda()
        
        # sourcePad=nn.functional.pad(sourceIn,(pad ,pad,pad ,pad),mode='reflect')
        
        
        
        denoised_img = scM(sourceIn).detach()
        
        out = denoised_img.squeeze().cpu().numpy()
        
        out_r= out[pad:-pad,pad:-pad]
        # dout_imgs.append(out )
        tempimg.append(out_r)
        hImg = np.concatenate(tempimg,axis=1)
        
                
vImg.append(hImg)
dout_imgs = np.concatenate(vImg,axis=0)


d_img= (dout_imgs/1000)* inMax

d_img=np.array(d_img).astype(np.uint16)


d_img.tofile("test0223_BonTech_3_Chest-phantom_110Lp-grid_3072x3072.raw")

#opencv test

gbImg= cv2.GaussianBlur(imgArray,(7,7),0)

ret, th = cv2.threshold(gbImg, 3000, 40000, cv2.THRESH_BINARY_INV)

th_=th.astype(np.uint8)
thr=cv2.erode(th_,None,iterations=5)
thr=cv2.dilate(thr,None,iterations=5)

contours, heiarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)


src_c= cv2.cvtColor(thr,cv2.COLOR_GRAY2BGR)

for i in range(len(contours)):
    # 랜덤 색상 지정
    c = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    
    cv2.drawContours(src_c, [contours[i]], 0, c, 2)
    # cv2.drawContours(dst, contours, idx, c, 2, cv2.LINE_8, hier)
    
    # 0번째 계층만 그리기. 하지만 hier 계층 정보를 입력했기 때문에 모든 외곽선에 그림을 그립니다.
    # 계층 정보를 입력 안하면 0번 계층만 그립니다.
    # idx = hier[0, idx, 0]

# for i in range(len(contours)):
#     cv2.drawContours(src_c, [contours[i]], 0, (0, 0, 255), 2)
#     cv2.putText(src_c, str(i), tuple(contours[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
#     print(i, heiarchy[0][i])
#     # cv2.imshow("src", src_c)
#     # cv2.waitKey(0)

mask = np.zeros(src_c.shape[:2], dtype="uint8")

mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

# cv2.drawContours(mask, [contours[18]], 0, (128,128,0), 2)
a=cv2.fillPoly(mask,[contours[18]],(128,128,0))


plt.imshow(a)

#%% Contour
import os
import cv2

loadFileName = "./test0222_BonTech_3_Chest-phantom_110Lp-grid_3072x3072.raw"

# imgArray = np.fromfile(loadFileName, dtype=np.float32).reshape(256,256)

fileName , file_ext = os.path.splitext(loadFileName)
file_sep = fileName.split("_")

img_f_size=file_sep[-1]

iw,ih=img_f_size.split("x")

img_w=int(iw)
img_h=int(ih)

imgArray = np.fromfile(loadFileName, dtype=np.uint16).reshape(img_w,img_h)



ret, th = cv2.threshold(imgArray, 3000, 40000, cv2.THRESH_BINARY_INV)

th_= th.astype(np.uint8)
# 컨투어 찾기와 그리기 ---②
contours, heiarchy = cv2.findContours(th_, cv2.RETR_EXTERNAL, \
                                         cv2.CHAIN_APPROX_SIMPLE)


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
                                # "--load-ckpt","../ckpts/grid-1759/n2n-epoch120-0.00099.pt",
                                # "--load-ckpt","../ckpts/grid-1058/n2n-epoch18-0.00068.pt",
                                # "--load-ckpt","../saved/grid-clean-210205/n2n-epoch996-2.95751.pt",
                                # "--load-ckpt","../saved/grid-clean-210215/n2n-epoch997-8.27363.pt",
                                # "--load-ckpt","../saved/grid-clean-210217/n2n-epoch993-5.26789.pt",
                                # "--load-ckpt","../saved/grid-clean-210218_test/n2n-epoch10-138.38759.pt",
                                # "--load-ckpt","../saved/grid-clean-210217/n2n-epoch993-5.26789.pt",
                                # "--load-ckpt","../saved/grid-clean-210218/n2n-epoch999-2.76057.pt",
                                "--load-ckpt","../saved/grid-clean-210312/n2n-epoch999-64.86539.pt",
                                
                                
                                
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
loadFileName = "./testImgs/BonTech_3_Chest-phantom_110Lp-grid_3072x3072.raw"
# loadFileName = "./testImgs/Before_grid_suppression_chest_phantom_3072x3072.raw"

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

imgArray = np.fromfile(loadFileName, dtype=np.uint16).reshape(img_w,img_h)

inMax = imgArray.max()
devicetype = "cuda"
# devicetype = "cpu"

timeS = timeit.default_timer()
d_imgO=n2n.testOneFileCrop(imgArray,img_w, img_h,256,16,devicetype )

timeE =timeit.default_timer()
print("run:: %f sec "%(timeE-timeS))

d_img= (d_imgO/1000)* inMax

d_img=np.array(d_img).astype(np.uint16)


d_img.tofile("./testOutImage/test0303_gpu_BonTech_3_Chest-phantom_110Lp-grid_3072x3072.raw")
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
# An instance of your model.
model = torchvision.models.resnet18()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)


#%%
save_sm_fileName= "ts_load_test_0.pt"
output = traced_script_module(torch.ones(1, 3, 224, 224))

print(output[0,:5])

traced_script_module.save(save_sm_fileName)

#%% load test
sm_load=torch.jit.load(save_sm_fileName)

output_2 = sm_load(torch.ones(1, 3, 224, 224))

print(output_2[0,:5])

#%% DCM load 

import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file
import os
import glob
from sys import platform
import numpy as np



# root_dir = f"D:\\10_jpi\\10_Project\\18_Grid_suppresion\chestImage_forGrid_raw_and_dcm\Chest sample images"
# root_dir = f"D:\\10_jpi\\10_Project\\20_DL\\DL_Grid_suppresion_Test\\noise2noise-pytorch\\data\\train_grid"
root_dir = f"D:\\10_jpi\\10_Project\\18_Grid_suppresion\\GridSupp_GLS_Test Sample"
imgList = os.listdir(root_dir)

imgName = imgList[0]
img_path = os.path.join(root_dir, imgName)
file_Name,file_ext=os.path.splitext(imgName)

size="".join(file_Name.split('_')[-1:]).split('x')

# if file_ext.lower() == ".dcm":
#     ds = dcmread(img_path )
#     npImg = ds.pixel_array



for imgName in imgList:
    img_path = os.path.join(root_dir, imgName)
    file_Name,file_ext=os.path.splitext(imgName)
    if file_ext.lower() == ".dcm":
        ds = dcmread(img_path )
        
        npImg = ds.pixel_array
        
        wFileName= root_dir+"\\"+file_Name + ".raw"
        npImg.tofile(wFileName)       
        
        print(f"writefile:{wFileName}")
        

# # fpath = get_testdata_file('CT_small.dcm')
# # fpath = '1417_103_r8_120cm_Long_Chest_1.dcm'
# fpath = 'Before grdi suppression chest phantom.dcm'
# ds = dcmread(fpath)

# # Normal mode:
# print()
# print(f"File path........: {fpath}")
# print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
# print()

# pat_name = ds.PatientName
# display_name = pat_name.family_name + ", " + pat_name.given_name
# print(f"Patient's Name...: {display_name}")
# print(f"Patient ID.......: {ds.PatientID}")
# print(f"Modality.........: {ds.Modality}")
# print(f"Study Date.......: {ds.StudyDate}")
# print(f"Image size.......: {ds.Rows} x {ds.Columns}")
# print(f"Pixel Spacing....: {ds.PixelSpacing}")

# # use .get() if not sure the item exists, and want a default value if missing
# print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

# # plot the image using matplotlib
# plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
# plt.show()

# imgShape = ds.pixel_array.shape


#%% find the sub directory dcm file and save the raw file


import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file
import os
import glob
from sys import platform
import numpy as np



# root_dir = f"D:\\10_jpi\\10_Project\\18_Grid_suppresion\chestImage_forGrid_raw_and_dcm\Chest sample images"
# root_dir = f"D:\\10_jpi\\10_Project\\20_DL\\DL_Grid_suppresion_Test\\noise2noise-pytorch\\data\\train_grid"
root_dir = f"D:\\10_jpi\\10_Project\\18_Grid_suppresion\\GridSupp_GLS_Test Sample"
imgList = os.listdir(root_dir)

fileList = glob.glob(root_dir+"\\**",recursive=True)


for imgName in fileList:
    # print(f"writefile:{imgName}")
    
    # img_path = os.path.join(root_dir, imgName)
    file_Name,file_ext=os.path.splitext(imgName)
    if file_ext.lower() == ".dcm":
        ds = dcmread(imgName )
        
        npImg = ds.pixel_array
        
        wfname = f"{file_Name}_{ds.Columns}x{ds.Rows}.raw"
        npImg.tofile(wfname )       
        
        print(f"writefile:{wfname}")


