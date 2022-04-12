# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:36:41 2021

@author: alexany
"""

from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
# 
import skimage.io
from cellpose import models
import os
import sys

import torch

import tifffile

def normalize99(img):
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

# sigmoid-linear unit
def SILIU(x,x0=0.0,width=0.1,p=6): # x presumed numpy array
    z = np.power((x-x0)/width,p)
    z = 1./(1.+np.exp(-z))
    return x*np.power(z,p)
######

# print(f"Arguments count: {len(sys.argv)}")
# for i, arg in enumerate(sys.argv):
#     print(f"Argument {i:>6}: {arg}")

ARGUMENTS = (sys.argv[1]);

ARGUMENTS = ARGUMENTS.split(':');

DSTDIR = ARGUMENTS[0]
fullfilename = ARGUMENTS[1]
MODE = ARGUMENTS[2]
MODELTYPE = ARGUMENTS[3]
CELLPROB_THRESHOLD = float(ARGUMENTS[4])
FLOW_THRESHOLD = float(ARGUMENTS[5])
RESAMPLE = ARGUMENTS[6]=='True'
DIAMETER = float(ARGUMENTS[7])
GPU = ARGUMENTS[8]=='True'
space_replacing_template = ARGUMENTS[9]

fullfilename = fullfilename.replace(space_replacing_template," ")

print(MODELTYPE)
print('image file exists - ' + str(os.path.isfile(fullfilename)))

if torch.cuda.is_available():
    print('using GPU')
else:
    print('not using GPU')
               
# DEFINE CELLPOSE MODEL
model = models.Cellpose(gpu=GPU, model_type=MODELTYPE)    

with tifffile.TiffFile(fullfilename) as tif:
    imgs = []
    for k in list(range(len(tif.pages))):
        imgs.append(tif.pages[k].asarray())
        
    n_frames = len(imgs)                
    w,h = np.shape(np.array(imgs[0]))
        
    sgm = np.zeros((n_frames,w,h),dtype=np.float32)
                
    for j in range(n_frames):
        print(j)        
        img = np.array(imgs[j])
        #
        if MODE=='DPC':
            # apply thresholding and weak noise suppression
            img[img<0]=0
            img = SILIU(img,width=0.12,p=6)
        #
        img = normalize99(img) 
        # 
        diam1 = DIAMETER
        masks, flows, styles, diams = model.eval(img, 
        diameter = diam1, 
        cellprob_threshold = CELLPROB_THRESHOLD,
        flow_threshold = FLOW_THRESHOLD,
        resample = RESAMPLE,
        channels=[0,0])
        # 
        sgm[j,:,:] = masks
    
# save results
split_filename = fullfilename.split(os.path.sep)
filename = split_filename[len(split_filename)-1]
#
save_fname = DSTDIR + os.path.sep + filename
skimage.io.imsave(save_fname, sgm, plugin='tifffile')

print('completed!')