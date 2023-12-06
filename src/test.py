#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import dataset
from FRINet_deit import FRINet

class Test(object):
    def __init__(self, Dataset, Network, path, model):
        ## dataset
        self.model  = './snapshot/'+model
        self.cfg    = Dataset.Config(datapath=path, snapshot=self.model, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
    def save(self):
        res=[]
        with torch.no_grad():
            for image, hf, mask, shape, name in self.loader:
                image = image.cuda().float()
                mask  = mask[0].cpu().numpy()*255
                hf    = hf.cuda().float()
                start = time.time()
                p = self.net(image,hf)
                end = time.time()
                res.append(end-start)
                out_resize   = F.interpolate(p[0],size=shape, mode='bilinear')
                pred   = torch.sigmoid(out_resize[0,0])
                pred  = (pred*255).cpu().numpy()
                head  = '../result/'+self.model+'/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))

        time_sum=0
        for i in res:
            time_sum += i
        print("FPS: %f "%(1.0/(time_sum/len(res))))

if __name__=='__main__':
    dir = '../data/TestDataset/'
    for path in ['COD10K','NC4K','CAMO']:
            t = Test(dataset,FRINet, dir+path,'model-100')
            t.save()
