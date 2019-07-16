# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:19:19 2019

@author: Effie
"""
import os 
import cv2
import random
import numpy as np
def RandomRotate(gray):
    (h,w)=gray.shape[:2]
    center=(w/2,h/2)
    angle=90*random.randint(0,4)
    scale=1.0
    M=cv2.getRotationMatrix2D(center,angle,scale)
    gray=cv2.warpAffine(gray,M,(h,w))
    return gray

def RandomScale(gray):
    scale_percent=random.randint(5,95)
    width=int(gray.shape[1]*scale_percent/100)
    height=int(gray.shape[0]*scale_percent/100)
    dim=(width,height)
    gray=cv2.resize(gray,dim,interpolation=cv2.INTER_AREA)
    return gray

def RandomCrop(gray):
    (h,w)=gray.shape[:2]
    hran=random.randint(0,int(h/2))
    wran=random.randint(0,int(w/2))
    gray=gray[0:int(h/2+hran),0:int(w/2+wran)]
    return gray

def RandomNoise(gray):
    noise=np.random.normal(0,2,(gray.shape))
    gray=gray+noise
    return gray


location='/home/jiahui/Desktop/Zhiwu/'
Igs=[];lb=[];lbi=[];
cate=os.listdir(location)
for i in cate: #Obtain the list of folders : seg_train or seg_pred or seg_test
    path=location+i; 
    L=os.listdir(path)
    for j in L:
           imgpath=path+r'/'+j; #Create the path for pircture j access
           gray=cv2.imread(imgpath,1) #read the images
           grayRotate=RandomRotate(gray)
           grayCrop=RandomCrop(gray)
           grayScale=RandomScale(gray)
           grayNoise=RandomNoise(gray)
           cv2.imwrite(location+i+'/'+j+'d.jpeg',grayNoise)
           cv2.imwrite(location+i+'/'+j+'c.jpeg',grayCrop)
           cv2.imwrite(location+i+'/'+j+'b.jpeg',grayScale)
           cv2.imwrite(location+i+'/'+j+'a.jpeg',grayRotate)
