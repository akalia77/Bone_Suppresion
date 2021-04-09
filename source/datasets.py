#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils import load_hdr_as_tensor

import os
import glob
import cv2
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw
# import OpenEXR

from argparse import ArgumentParser

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpImg
from Test_grid import createImageWidthGrid


def load_dataset(root_dir, redux, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)

    # Instantiate appropriate dataset class

    if params.noise_type == 'grid':
        dataset =GridDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)
    elif params.noise_type == 'bone':
        dataset =BoneDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)
    else:
        dataset = NoisyDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size

        cropped_imgs = []

        if w < self.crop_size or h < self.crop_size:
            # assert w >= self.crop_size and h >= self.crop_size, \
            #     f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'

            print('waring: Crop size: {self.crop_size}, Image size: ({w}, {h})')
            print("resize =====>")
            i = 0
            j = 0

            # if min(w, h) < self.crop_size:
            #     img = tvF.resize(img_list[0], (self.crop_size, self.crop_size))


        else:
            i = np.random.randint(0, h - self.crop_size + 1)
            j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs
    
    


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)


class NoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, redux, crop_size, clean_targets=False,
        noise_dist=('gaussian', 50.), seed=None):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        self.imgs = os.listdir(root_dir)
        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)


    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        w, h = img.size
        c = len(img.getbands())

        # Poisson distribution
        # It is unclear how the paper handles this. Poisson noise is not additive,
        # it is data dependent, meaning that adding sampled valued from a Poisson
        # will change the image intensity...
        if self.noise_type == 'poisson':
            noise = np.random.poisson(img)
            noise_img = img + noise
            noise_img = 255 * (noise_img / np.amax(noise_img))

        # Normal distribution (default)
        else:
            if self.seed:
                std = self.noise_param
            else:
                std = np.random.uniform(0, self.noise_param)
            noise = np.random.normal(0, std, (h, w, c))

            # Add noise and clip
            noise_img = np.array(img) + noise

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)



    def _add_text_overlay(self, img):
        """Adds text overlay to images."""

        assert self.noise_param < 1, 'Text parameter is an occupancy probability'

        w, h = img.size
        c = len(img.getbands())

        # Choose font and get ready to draw
        if platform == 'linux':
            serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'C:\Windows\Fonts\Times New Roman.ttf'
        text_img = img.copy()
        text_draw = ImageDraw.Draw(text_img)

        # Text binary mask to compute occupancy efficiently
        w, h = img.size
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        # Random occupancy in range [0, p]
        if self.seed:
            random.seed(self.seed)
            max_occupancy = self.noise_param
        else:
            max_occupancy = np.random.uniform(0, self.noise_param)
        def get_occupancy(x):
            y = np.array(x, dtype=np.uint8)
            return np.sum(y) / y.size

        # Add text overlay by choosing random text, length, color and position
        while 1:
            # font = ImageFont.truetype(serif, np.random.randint(16, 21))
            
            font = ImageFont.load_default()
            length = np.random.randint(10, 25)
            chars = ''.join(random.choice(ascii_letters) for i in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)

            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break

        return text_img


    def _corrupt(self, img):
        """Corrupts images (Gaussian, Poisson, or text overlay)."""

        if self.noise_type in ['gaussian', 'poisson']:
            return self._add_noise(img)
        # elif self.noise_type == 'grid':
        #     return self._add_grid(img)
        elif self.noise_type == 'text':
            return self._add_text_overlay(img)
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL imagePP
        img_path = os.path.join(self.root_dir, self.imgs[index])
        file_Name,file_ext=os.path.splitext(img_path)
        if file_ext.lower() == ".raw":
            r_w= 3072
            r_h= 3072
            imgArray = np.fromfile(img_path, dtype=np.uint16).reshape(r_w,r_h)
            img =Image.fromarray(imgArray)
            
        else:
            img =  Image.open(img_path).convert('RGB')


        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop([img])[0]

        # Corrupt source image
        tmp = self._corrupt(img)
        source = tvF.to_tensor(self._corrupt(img))

        # Corrupt target image, but not when clean targets are requested
        if self.clean_targets:
            target = tvF.to_tensor(img)
        else:
            target = tvF.to_tensor(self._corrupt(img))

        # source = torch.unsqueeze(source,0) // test
        return source, target

class BoneDataset(AbstractDataset):
    def __init__(self, root_dir, redux, crop_size, clean_targets=False,
                 noise_dist=('gaussian', 50.),seed=None):
        """Initializes noisy image dataset."""

        super(BoneDataset, self).__init__(root_dir, redux, crop_size, clean_targets)
        
        
        src_dir=os.path.join(self.root_dir, "source")
        trg_dir=os.path.join(self.root_dir, "target")
        self.srcDir= src_dir
        self.trgDir= trg_dir

        

        self.imgs = os.listdir(src_dir)
        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        
    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

            # Load PIL imagePP
        img_name = self.imgs[index]
        img_path_src = os.path.join(self.srcDir, img_name)
        img_path_trg = os.path.join(self.trgDir, img_name)
        
        
        
        srcImg =  Image.open(img_path_src).convert('L')
        trgImg =  Image.open(img_path_trg).convert('L')
        
        
        # srcImg = cv2.imread(img_path_src,cv2.IMREAD_GRAYSCALE)       

        # trgImg = cv2.imread(img_path_trg,cv2.IMREAD_GRAYSCALE)
        
        # # srcImg =  mpImg.imread(img_path_src)[:,:,0]*10000
        # # trgImg =  mpImg.imread(img_path_trg)*10000
        r= np.random.randint(-10,8)
        out_s = tvF.rotate(srcImg,r)
        out_t = tvF.rotate(trgImg,r)
        
        rSize=np.random.randint(15)
        outSize = (srcImg.size[1]-rSize,srcImg.size[0]-rSize)
        i, j, h, w = transforms.RandomCrop.get_params(out_s, output_size=outSize)

        # print(f"rotation: {r}, cropSize: {rSize}")
        
        out_s= tvF.crop(out_s, i, j, h, w)
        out_t= tvF.crop(out_t, i, j, h, w)
        
        # f_resize= transforms.Resize(self.crop_size)

        # srcImg = cv2.resize(srcImg,(self.crop_size,self.crop_size))
        # trgImg = cv2.resize(trgImg,(self.crop_size,self.crop_size))

        # srcImg = srcImg,(self.crop_size,self.crop_size))
        # trgImg = trgImg,(self.crop_size,self.crop_size))
        
        
        # srcImg = f_resize(out_s)
        # trgImg = f_resize(out_t)
        
        # float image 
        srcImg=out_s.resize((self.crop_size,self.crop_size),resample=Image.BICUBIC)
        trgImg=out_t.resize((self.crop_size,self.crop_size),resample=Image.BICUBIC)
        
        srcImg = (srcImg/np.amax(srcImg) * 1000).astype(np.float32)
        trgImg = (trgImg/np.amax(trgImg) * 1000).astype(np.float32)
        
        source = tvF.to_tensor(srcImg)
        target = tvF.to_tensor(trgImg)
        
        
        return source, target
            
            

class GridDataset(AbstractDataset):
    """Class for dealing with Grid images."""

    def __init__(self, root_dir, redux, crop_size, clean_targets=False,
        noise_dist=('gaussian', 50.), seed=None):
        """Initializes noisy image dataset."""

        super(GridDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        self.imgs = os.listdir(root_dir)
        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
    def _add_grid(self,img):
        
        # if type(img).__module__ == Image.__name__:
        #     print("InmoduleType == Image")
        # else:
        #     print("InmoduleType != Image")

        w, h = img.size
        
        angle_val = np.random.uniform(0, 3)
        isHorizontal = np.random.randint(0,2) 
        # isHorizontal = 0
        
        t1_rate= 2 # with add random value
        t2_rate = -3
        
        # org test value 
        # t1_with = 2
        # tw_total = 4.2


        t1_with =np.random.uniform(low=1.5,high=5)
        tw_total =t1_with + np.random.uniform(low=1.5,high=3)
        
        # t1_rate= 20 * np.random.binomial(10,0.5)/10
        # t2_rate = -10* np.random.binomial(10,0.7)/10


        # t1_with = 2*np.random.uniform(0.4, 2)
        # tw_total = t1_with + np.random.uniform(0.3, 1)
        
        # print("T1W: " + str(t1_with) +"TW: " + str(tw_total))

        imgNp = np.array(img).astype(np.float32)
        
        imgType = imgNp.dtype
        
        # imgNp = imgNp/imgNp.max()        
        
        retImg = createImageWidthGrid(imgNp  ,w,h,
                                      t1_rate,t2_rate,t1_with,tw_total,
                                      angle_val,isHorizontal)
        # if retImg.max() !=0:
        #     retImg = retImg/retImg.max()
        # retImgOut = retImg.astype(np.float32)         

        outImg = Image.fromarray(retImg)
        

        return outImg 


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

            # Load PIL imagePP
        img_path = os.path.join(self.root_dir, self.imgs[index])
        
        # print(f"load::{img_path}")
        
        file_Name,file_ext=os.path.splitext(img_path)
        if file_ext.lower() == ".raw":
            size="".join(file_Name.lower().split('_')[-1:]).split('x')
            
            # r_w= 3072
            # r_h= 3072
            r_h = int(size[0])
            r_w = int(size[1])
            imgArray = np.fromfile(img_path, dtype=np.uint16).reshape(r_w,r_h).copy()
            imgArray = imgArray/imgArray.max()*1000
            # imgArray = imgArray/imgArray.max()

            img =Image.fromarray(imgArray)
            
            
        else:
            img =  Image.open(img_path).convert('L')
        
        # img_np=np.squeeze(np.array(img)).astype(np.uint16)*100 
        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop([img])[0]

        # Corrupt source image
        # tmp = self._add_grid(img)
        # tmp = np.array(self._add_grid(img))
        # tmp_np.astype(np.uint16).tofile(f"testImg_jpg_grid_{tmp_np.shape[1]}_x_{tmp_np.shape[0]}_uint16.raw")
        source = tvF.to_tensor(self._add_grid(img))
        

        # Corrupt target image, but not when clean targets are requested
        if self.clean_targets:
            target = tvF.to_tensor(img)
        else:
            target = tvF.to_tensor(self._add_grid(img))

        # source = torch.unsqueeze(source,0) // test
        return source, target



def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='../data')    
    parser.add_argument('--show-output', help='pop up window to display outputs', default=0, type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'grid','poisson', 'text', 'mc','bone'], default='gaussian', type=str)
    parser.add_argument('-v', '--noise-param', help='noise parameter (e.g. sigma for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=256, type=int)
    parser.add_argument('-b', '--bacth-size', help='image batch size', default=64, type=int)
    args = parser.parse_args(
                            [
                            # "--data","../data/test",
                            "--data","../datasets/train_bone",
                            # "--load-ckpt","../ckpts/gaussian/n2n-gaussian.pt",
                            # "--noise-type","gaussian",                            
                            # "--noise-type","grid",
                            "--noise-type","bone",
                            # "--noise-param","50",
                            "--noise-param","0.4",
                            "--crop-size","256",
                            "--show-output","3",
                            "--cuda",
                            ])

    # return parser.parse_args()
    return args



if __name__ == '__main__':
    params = parse_args()

    params.redux = False
    params.clean_targets = True
    params.batch_size = 1

    # test_loader = load_dataset(params.data, 0, params, shuffled=True, single=True)
    test_loader = load_dataset(params.data, 0, params, shuffled=True)

    for batch_idx, (source, target) in enumerate(test_loader):
            # Only do first <show> images
            # pilImage_source=tvF.to_pil_image(torch.squeeze( source))
            # plt.imshow(pilImage_source)
            # plt.pause(1)
            # print('ide: ' + str(batch_idx))
            
            for idx,s_tensor in enumerate(source):
                img_s = np.squeeze( s_tensor).cpu()
                img_t = np.squeeze( target[idx]).cpu()
                dspImg = np.concatenate((img_s,img_t),axis=1)
                plt.imshow(dspImg)
                plt.pause(1)
                print('idx: ' + str(idx))


    print('test done !')
            



    