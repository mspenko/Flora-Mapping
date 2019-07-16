#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:28:09 2019

@author: jiahui
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:52:42 2019

@author: Effie
"""

from keras.layers import Flatten
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
from PIL import Image
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.models import load_model
import time

model=load_model('modelres.h5')
location='/home/jiahui/Desktop/insta_test/30/'
cade=os.listdir(location)
print(len(cade))
ib=[];lbi=[];
all_images=[]
for i in cade: 
    path=location+i;
    gray=cv2.imread(path,1)
    #HSV=cv2.cvtColor(gray,cv2.COLOR_BGR2HSV)
    #H,S,V=cv2.split(HSV)
    #LowerBlue=np.array([35,43,46])
    #LowerBlue=np.array([35,43,46])
    #UpperBlue=np.array([99,255,255])
    #mask=cv2.inRange(HSV,LowerBlue,UpperBlue)
    #gray=cv2.bitwise_and(gray,gray,mask=mask)
    gray=Image.fromarray(gray,mode=None)
    gray=np.array(gray.resize((150,150)))
    all_images.append(gray)
    ib.append(i)

shuffle(all_images,ib,random_state=81743)
x_pred=np.array(all_images)/255

cate=os.listdir('/home/jiahui/Desktop/insta/')
print(cate)
o=0;z=0;
j=0;b=0;k=0;e=0;f=0;g=0;
index=0;
while index<len(x_pred):
    img=(np.expand_dims(x_pred[index],0))
    predictions=model.predict(img)
    pred_img=np.argmax(predictions[0])
    if pred_img==20:o=o+1;
    elif pred_img==14:j=j+1;
    elif pred_img==15:k=k+1; 
    elif pred_img==16:e=e+1;
    elif pred_img==17:f=f+1;
    elif pred_img==18:g=g+1;
    elif pred_img==19:b=b+1;       
    index = index+1


