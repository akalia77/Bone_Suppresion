#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from datasets import load_dataset
from noise2noise import Noise2Noise
from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter

#  https://github.com/joeylitalien/noise2noise-pytorch


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/valid')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=500, type=int)
    parser.add_argument('-ts', '--train-size', help='size of train dataset', type=int)
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', type=int)

    # Training hyperparameters
    # parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=4, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', 
                        choices=['l1', 'l2', 'hdr','ssim','ms-ssim','mse-ssim'], default='l1', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')
    parser.add_argument('--load-ckpt', help='load model checkpoint')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'grid','poisson', 'text', 'mc','bone'], default='gaussian', type=str)
    parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=128, type=int)
    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')

    args = parser.parse_args(
                            [
                            # "--train-dir","../data/train_grid","--train-size","25",
                            "--train-dir","../datasets/train_bone","--train-size","100",
                            "--valid-dir","../datasets/valid_bone","--valid-size","20",
                                
                            # "--train-dir","../data/train","--train-size","30000",
                            # "--valid-dir","../data/valid","--valid-size","100",
                            "--nb-epochs","1000",
                            
                            "--ckpt-save-path","../ckpts",
                            "--batch-size","5",
                            "--report-interval","20",
                            # "--batch-size","1",
                            # "--report-interval","50",

                            # "--learning-rate" , "0.001"

                            
                            # "--load-ckpt","../saved/bone-clean-210331-1723/n2n-epoch999-1.01681.pt",
                            
                            # "--loss","ssim",
                            # "--loss","ms-ssim",
                            "--loss","mse-ssim",
                            
                            "--noise-type","bone",
                            "--noise-param","50",                            
                            # "--noise-type","gaussian",
                            # "--noise-param","50",
                            # "--noise-type","text",
                            # "--noise-param","0.6",
                            '--clean-targets',
                            # "--crop-size","128",
                            "--crop-size","1024",
                            "--plot-stats",
                            "--cuda",
                            ])
    return args


if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Parse training parameters
    params = parse_args()

    # Train/valid datasets
    train_loader = load_dataset(params.train_dir, params.train_size, params, shuffled=True)
    valid_loader = load_dataset(params.valid_dir, params.valid_size, params, shuffled=False)

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    
    if None != params.load_ckpt:
        n2n.load_model(params.load_ckpt)
    else:
        print("\nload check file is NULL!! \n ")

    n2n.train(train_loader, valid_loader)

    # writer = SummaryWriter('runs/fashion_mnist_experiment_1')
