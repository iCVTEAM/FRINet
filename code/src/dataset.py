#!/usr/bin/python3
#coding=utf-8
import random
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
from PIL import ImageEnhance,Image
########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask,edge=None):
        image = (image - self.mean)/self.std
        mask /= 255
        if edge is not None:
            edge /= 255
            return image,mask,edge
        return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask,edge=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if edge is not None:
            edge  = cv2.resize( edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image,mask,edge
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        return image, mask

########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

def laplacian(img):
    size = img.shape[0]
    img_down_2x = cv2.resize(img,(size//2,size//2))
    img_down_4x = cv2.resize(img,(size//4,size//4))
    img_down_8x = cv2.resize(img,(size//8,size//8))
    img_down_16x = cv2.resize(img,(size//16,size//16))

    img_down_2x_up = cv2.resize(img_down_2x,(size,size))
    img_down_4x_up = cv2.resize(img_down_4x,(size,size))
    img_down_8x_up = cv2.resize(img_down_8x,(size,size))
    img_down_16x_up = cv2.resize(img_down_16x,(size,size))

    x2 = img-img_down_2x_up
    x4 = img-img_down_4x_up
    x8 = img-img_down_8x_up
    x16= img-img_down_16x_up

    return np.concatenate((x2,x4,x8,x16),axis=2)

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return np.array(image)

def rgb_loader( path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
def randomRotation(image, label,edge=None):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        if edge is not None:
            edge = edge.rotate(random_angle, mode)
    if edge is not None:
        return image,label,edge
    return image, label
def random_flip(img, label,edge=None):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if edge is not None:
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    if edge is not None:
        return img,label,edge
    return img, label
def PIL_randomCrop(image, label,edge=None):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    if edge is not None:
        return image.crop(random_region), label.crop(random_region),edge.crop(random_region)
    return image.crop(random_region), label.crop(random_region)
########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.resize     = Resize(448, 448)
        self.resizet    = Resize(448, 448)
        self.totensor   = ToTensor()
        self.samples    = []
        lst = glob.glob(cfg.datapath +'/mask/'+'*.png')

        for each in lst:
            img_name = each.split("/")[-1]
            img_name = img_name.split(".")[0]
            self.samples.append(img_name)
    def __getitem__(self, idx):
        name  = self.samples[idx]
        tig='.jpg'
        image = rgb_loader(self.cfg.datapath+'/image/'+name+tig)
        mask  = binary_loader(self.cfg.datapath+'/mask/' +name+'.png')
        edge  = binary_loader(self.cfg.datapath+'/image/'+name+tig)
        if self.cfg.mode=='train':
            image,mask,edge = random_flip(image,mask,edge=edge)
            image,mask,edge = PIL_randomCrop(image,mask,edge=edge)
            image,mask,edge = randomRotation(image,mask,edge=edge)
            image = colorEnhance(image)
            image = np.asarray(image).astype(np.float32)
            mask = np.asarray(mask).astype(np.float32)
            edge = np.asarray(edge).astype(np.float32)
            image, mask, edge = self.resize(image, mask, edge)
            image, mask, edge = self.normalize(image, mask, edge)
            hf = laplacian(image)
            return image.copy(), hf.copy(),mask.copy(),edge.copy()

        else:
            image = np.asarray(image).astype(np.float32)
            mask = np.asarray(mask).astype(np.float32)
            shape = mask.shape #
            image, mask = self.normalize(image, mask)
            image, mask = self.resizet(image, mask)
            hf   = laplacian(image)
            image, mask = self.totensor(image, mask)
            hf = torch.from_numpy(hf).permute(2, 0, 1)
            return image, hf, mask, shape, name

    def collate(self, batch):
        size = [256,320,384,448,512][np.random.randint(0, 5)]
        image , hf , mask, edge = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            hf[i]    = cv2.resize(hf[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            edge[i]  = cv2.resize(edge[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        hf = torch.from_numpy(np.stack(hf, axis=0)).permute(0,3,1,2)
        mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        edge  = torch.from_numpy(np.stack(edge, axis=0)).unsqueeze(1)
        return image, hf,mask,edge

    def __len__(self):
        return len(self.samples)

