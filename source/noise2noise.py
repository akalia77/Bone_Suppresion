#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import cv2
import numpy as np 
from torch.optim import Adam, lr_scheduler
import torchvision.transforms.functional as tvF
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim

# from unet import UNet
from unetTest import UNet
from utils import *

import os
import json
import timeit
from tensorboardX import SummaryWriter
import timeit
from torch.utils import mkldnn


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(MS_SSIM_Loss, self).forward(img1, img2) )

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()


    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print('Noise2Noise: Learning Image Restoration without Clean Data (Lethinen et al., 2018)')

        # Model (3x3=9 channels for Monte Carlo since it uses 3 HDR buffers)
        if self.p.noise_type == 'mc':
            self.is_mc = True            
            self.model = UNet(in_channels=9)
        elif self.p.noise_type =='grid':
            self.is_mc = False
            self.model = UNet(in_channels=1,out_channels=1)
        elif self.p.noise_type =='bone':
            self.is_mc = False
            self.model = UNet(in_channels=1,out_channels=1)
        else:
            self.is_mc = False
            self.model = UNet(in_channels=3)

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.p.nb_epochs/4, factor=0.5, verbose=True)

            # Loss function
            if self.p.loss == 'hdr':
                assert self.is_mc, 'Using HDR loss on non Monte Carlo images'
                self.loss = HDRLoss()
            elif self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            elif self.p.loss == 'ssim':                
                self.loss  = SSIM_Loss(data_range=1.0, size_average=True, channel=1)
            elif self.p.loss == 'ms-ssim':                
                self.loss   = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=1)
            else:
                self.loss = nn.L1Loss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        
        # self.use_cuda = False
        print('use_cuda is : %s'%( str( self.use_cuda)))
        
        
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()


    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            if self.p.clean_targets:
                ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-clean-%y%m%d-%H%M}'
            else:
                ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-%y%m%d-%H%M}'
            if self.p.ckpt_overwrite:
                if self.p.clean_targets:
                    ckpt_dir_name = f'{self.p.noise_type}-clean'
                else:
                    ckpt_dir_name = self.p.noise_type

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/n2n-{}.pt'.format(self.ckpt_dir, self.p.noise_type)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/n2n-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/n2n-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)


    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
            print('modelLoad on : cuda')
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location=torch.device('cpu')))
            print('modelLoad on : cpu')
            
        


    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr ,imgSrc,imgDe,imgClean = self.eval(valid_loader,epoch)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)

        self.saveImage(epoch,valid_loss,imgSrc,imgDe,imgClean)

        # Plot stats
        if self.p.plot_stats:
            loss_str = f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')

    def testOnefile(self,inPath,w,h):
        
        self.model.train(False)
        
        # loadFileName = "testImg_jpg_grid_480_x640_uint16.raw"
        loadFileName = inPath
        
        imgArray = np.fromfile(loadFileName, dtype=np.uint16).reshape(w,h)
        
        # imgArray = imgArray/imgArray.max()
        img =Image.fromarray(imgArray)
        
        img = np.array(img).astype(np.float32)
        
        # img = img/img.max() # test 
        img = img/img.max()*1000 # org 
        
        source = tvF.to_tensor(img)
        
        source = torch.unsqueeze(source,0).cuda()
        
        denoised_img = self.model(source).detach()
        
        return denoised_img
    
    def padImg(img, inImg, b_size):
        
        inImg=np.arange(0,12).reshape(3,4)
        
        
        w=inImg.shape[1]
        h=inImg.shape[0]
        
        img=np.pad(inImg,(h+b_size*2,w+b_size*2),'edge')
        
        # img=np.zeros((h+b_size*2,w+b_size*2))
        
        # upArr=np.squeeze([inImg[:1,:]]*b_size)
        
        # img[:b_size,b_size:-b_size]=upArr
        
        # img[b_size:-b_size,b_size:-b_size]=inImg
        
        # downArr=np.squeeze([b[-1:,:]]*b_size)
        # img[-b_size:,b_size:-b_size]=downArr
        
        
        return img
        

    def testOneFileCrop(self,imgArray,w,h,c_size,b_size,devicetype):
        
        self.model.train(False)
        
        if devicetype == 'cpu':
            m_cpu= self.model.cpu()
            print("run:: cpu ")
        else:
            print("run:: cuda ")
        
        
        
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
        
        # model_mkldnn=mkldnn.to_mkldnn(self.model)
        
        
        
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
                
                # if self.use_cuda:
                if devicetype == "cuda":
                    sourceIn = torch.unsqueeze(source,0).cuda()
                    
                    ts= timeit.default_timer()
                    denoised_img = self.model(sourceIn).detach()
                    te= timeit.default_timer()
                    print("m_run:: %f sec "%(te-ts))
                    
                    out = denoised_img.squeeze().cpu().numpy()
                else:
                    # sourceIn = torch.unsqueeze(source,0).to_mkldnn()
                    sourceIn = torch.unsqueeze(source,0).cpu()
                    
                    ts= timeit.default_timer()
                    
                    denoised_img = m_cpu(sourceIn).detach()
                    # denoised_img = self.model(sourceIn).detach()
                    # denoised_img = model_mkldnn(sourceIn).detach()
                    te= timeit.default_timer()
                    print("m_run:: %f sec "%(te-ts))
                    
                    out = denoised_img.squeeze().numpy()
                
                # sourcePad=nn.functional.pad(sourceIn,(pad ,pad,pad ,pad),mode='reflect')
                
                out_r= out[pad:-pad,pad:-pad]
                # dout_imgs.append(out )
                tempimg.append(out_r)
                hImg = np.concatenate(tempimg,axis=1)
                
                        
        vImg.append(hImg)
        dout_imgs = np.concatenate(vImg,axis=0)
        
        return dout_imgs
        
    def test(self, test_loader, show):
        """Evaluates denoiser on test set."""

        self.model.train(False)

        source_imgs = []
        denoised_imgs = []
        clean_imgs = []

        # Create directory for denoised images
        denoised_dir = os.path.dirname(self.p.data)
        save_path = os.path.join(denoised_dir, 'denoised')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        for batch_idx, (source, target) in enumerate(test_loader):
            # Only do first <show> images
            if show == 0 or batch_idx >= show:
                break

            # pilImage_source=tvF.to_pil_image(torch.squeeze( source))
            source_imgs.append(source)
            clean_imgs.append(target)

            if self.use_cuda:
                source = source.cuda()
            
            #test:: save the torch script module create 
            # torch_script_module = torch.jit.trace(self.model,source)
            # torch_script_module.save("n2n_torch_script_grid_grayscale_t0.pt")
            # torch_script_module_sm = torch.jit.script(self.model)

            #test::


            # Denoise
            time_start = timeit.default_timer()
            
            denoised_img = self.model(source).detach()

            time_elapsed = timeit.default_timer()-time_start
            print('Testing Process done! Total elapsed time: {} msec\n'.format(time_elapsed*1000))


            denoised_imgs.append(denoised_img)
            
        # Squeeze tensors
        source_imgs = [t.squeeze(0) for t in source_imgs]
        denoised_imgs = [t.squeeze(0) for t in denoised_imgs]
        clean_imgs = [t.squeeze(0) for t in clean_imgs]
        
        s_np=np.squeeze(source_imgs[0].cpu().numpy()*10000).astype(np.uint16)
        d_np=np.squeeze(denoised_imgs[0].cpu().numpy()*10000).astype(np.uint16)
        c_np= np.squeeze(clean_imgs [0].cpu().numpy()*10000).astype(np.uint16)
        
        
        # Create montage and save images
        print('Saving images and montages to: {}'.format(save_path))
        print('Saving raw images to: {}'.format(save_path))
        wfileName = save_path+f"\\test_source_{s_np.shape[1]}x{s_np.shape[0]}_uint16.raw"        
        s_np.tofile(wfileName )
        
        wfileName = save_path+f"\\test_denoise_{s_np.shape[1]}x{s_np.shape[0]}_uint16.raw"
        d_np.tofile(wfileName )
        
        wfileName = save_path+f"\\test_clear_{s_np.shape[1]}x{s_np.shape[0]}_uint16.raw"
        c_np.tofile(wfileName)
        
        print('Saving raw images to: {}'.format(wfileName ))
        
        for i in range(len(source_imgs)):
            img_name = test_loader.dataset.imgs[i]
            create_montage(img_name, self.p.noise_type, save_path,
                           source_imgs[i], denoised_imgs[i], clean_imgs[i], show)

    def saveImage(self,epoch,valid_loss, source_t, denoised_t, clean_t,):
        src= source_t.detach().cpu().numpy().squeeze()
        src_denoise= denoised_t.detach().cpu().numpy().squeeze()
        target= clean_t.detach().cpu().numpy().squeeze()
        
        diff = src_denoise - target

        # fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        fig, ax = plt.subplots(2, 2, figsize=(9, 9))
        # fig.canvas.set_window_title(img_name.capitalize()[:-4])

        # Bring tensors to CPU
        # source_t = source_t.cpu().narrow(0, 0, source_t.size()[0])
        # denoised_t = denoised_t.cpu()
        # clean_t = clean_t.cpu()
        
        # source = tvF.to_pil_image(source_t)
        # denoised = tvF.to_pil_image(torch.clamp(denoised_t, 0, 1))
        # clean = tvF.to_pil_image(clean_t)

        # Build image montage
        # psnr_vals = [psnr(source_t, clean_t), psnr(denoised_t, clean_t)]
        titles = ['Input',
                'Denoised',
                'Ground truth',
                'Diff']
        
        zipped = zip(titles, [src, src_denoise, target])
        for j, (title, img) in enumerate(zipped):
            ax[int(j/2),int(j%2)].imshow(img)
            ax[int(j/2),int(j%2)].set_title(title)
            ax[int(j/2),int(j%2)].axis('off')


        # Save to files
        # fname = os.path.splitext(img_name)[0]
        fname = "Out"
        saveImgName = f'{fname}-{epoch}-{valid_loss:.6f}-montage.png'
        # source.save(os.path.join(save_path, f'{fname}-{noise_type}-noisy.png'))
        # denoised.save(os.path.join(save_path, f'{fname}-{noise_type}-denoised.png'))
        fig.savefig(os.path.join(self.ckpt_dir, saveImgName ), bbox_inches='tight')
        
        print(f"saveImg: {saveImgName}")


        # plt.imshow(img)
        plt.close()





    def eval(self, valid_loader,epoch):
        """Evaluates denoiser on validation set."""

        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()

        rSrc =[]
        rde =[]
        rtarget =[]

        for batch_idx, (source, target) in enumerate(valid_loader):
            if self.use_cuda:
                source = source.cuda()
                target = target.cuda()

            # Denoise
            source_denoised = self.model(source)

            # Update loss
            loss = self.loss(source_denoised, target)
            loss_meter.update(loss.item())

            if batch_idx == 0:
                rSrc=source[0]            
                rde=source_denoised[0]            
                rtarget=target[0]            

            # Compute PSRN
            if self.is_mc:
                source_denoised = reinhard_tonemap(source_denoised)
            # TODO: Find a way to offload to GPU, and deal with uneven batch sizes
            for i in range(self.p.batch_size):
                source_denoised = source_denoised.cpu()
                target = target.cpu()
                
                psnr_meter.update(psnr(source_denoised[i], target[i]).item())

        # sumWriter = SummaryWriter()
        # self.saveImage(saveImg)


        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg



        # sumWriter.add_scalar('loss/loss_a',valid_loss,epoch)
        # sumWriter.add_scalar('psnr/psnr_a',psnr_avg,epoch)
        # sumWriter.add_image('img/epoch_img',saveImg,epoch)
        
        # sumWriter.close() ,rSrc,rSrc_de,rtarget
        
        return valid_loss, valid_time, psnr_avg ,rSrc,rde,rtarget

    def train(self, train_loader, valid_loader):
        """Trains denoiser on training set."""

        self.model.train(True)

        self._print_params()
        num_batches = len(train_loader)
        assert num_batches % self.p.report_interval == 0, (f'Report interval must divide total number::{self.p.report_interval} of batches::{num_batches }')

        # Dictionaries of tracked stats
        stats = {'noise_type': self.p.noise_type,
                 'noise_param': self.p.noise_param,
                 'train_loss': [],
                 'valid_loss': [],
                 'valid_psnr': []}

        # Main training loop
        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (source, target) in enumerate(train_loader):
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                # Denoise image
                source_denoised = self.model(source)

                loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                # b=target.detach().squeeze().cpu()
                # a=source_denoised.detach().squeeze().cpu()
                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.scheduler.step(loss.item())

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()
        
            # Epoch end, save and reset tracker
            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))


class HDRLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""

        super(HDRLoss, self).__init__()
        self._eps = eps


    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""

        loss = ((denoised - target) ** 2) / (denoised + self._eps) ** 2
        return torch.mean(loss.view(-1))
