# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
from PIL import Image
from keras.utils import to_categorical

import time


NAME="LEAF{}".format(int(time.time()))
tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME))

trainset='/home/jiahui/Desktop/insta/'
#Define a function for archive pictures from directories
def img(location):
    
    Igs=[];lb=[];lbi=[]; #Initialization some empty varaibles for storage
    #Igs:images; lb:name label; lbi:label intergers
    #Obtain the categories in seg_trian folder
    cate=os.listdir(location)

    for i in cate: #Obtain the list of folders : seg_train or seg_pred or seg_test
        path=location+i; 
        L=os.listdir(path) #Obtain the list of pictures in folders
        for j in L:
            imgpath=path+r'/'+j; #Create the path for pircture j access
            if j=='desktop.ini':print('ss')
            else:
                gray=cv2.imread(imgpath,1) #read the images
                #HSV=cv2.cvtColor(gray,cv2.COLOR_BGR2HSV)
                #H,S,V=cv2.split(HSV)
                #LowerBlue=np.array([35,43,46])
                #LowerBlue=np.array([35,43,46])
                #UpperBlue=np.array([99,255,255])
                #mask=cv2.inRange(HSV,LowerBlue,UpperBlue)
                #gray=cv2.bitwise_and(gray,gray,mask=mask)
                imgarr=Image.fromarray(gray,mode=None) #Convert images to array
                imgsize=np.array(imgarr.resize((180,180)))#onvert all the pictures to same graphical size
                Igs.append(imgsize) #Store the image information
                lb.append(i) #Store the name label information
    k=0;
    while k<len(cate):
        for i in lb: #Get the respective label index
            if i==cate[k]:lbi.append(k);
        k=k+1
    return shuffle(Igs,lb,lbi, random_state=92893829)
Igs, lb, lbi =img(trainset)
X_train=np.array(Igs)/255
Y_train=to_categorical(np.array(lbi)) #Categorize the Y_Train for CNN training model & 6 categories
lb=np.array(lb)
print(X_train.shape)
print(Y_train.shape)
LB=lb

#Plot out the images to check if the associations are correct
fig,ax = plot.subplots(5,5,figsize=(15,15))  # 5 col/rows, figure size is 15x15
for i in range(0,5,1):
    for j in range(0,5,1):
        rd= randint(0,len(X_train)) #get random images in the range of [0,length(X_train)]
        ax[i,j].imshow(X_train[rd])
        ax[i,j].set_title(lbi[rd])
        ax[i,j].axis('off') #'[i,j] represents the image location'
        
from keras.models import Model
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
import numpy as np
seed = 7
np.random.seed(seed)
def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x
# Inception Block
def Inception(x,nb_filter):
    branch1x1 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    branch3x3 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    branch3x3 = Conv2d_BN(branch3x3,nb_filter,(3,3), padding='same',strides=(1,1),name=None)
    branch5x5 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    branch5x5 = Conv2d_BN(branch5x5,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branchpool = Conv2d_BN(branchpool,nb_filter,(1,1),padding='same',strides=(1,1),name=None)
    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3) 
    return x

inpt = Input(shape=(180,180,3))
x = Conv2d_BN(inpt,64,(7,7),strides=(2,2),padding='same')
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
x = Conv2d_BN(x,192,(3,3),strides=(1,1),padding='same')
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
x = Inception(x,64)#256
x = Inception(x,120)#480
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
x = Inception(x,128)#512
x = Inception(x,128)
x = Inception(x,128)
x = Inception(x,132)#528
x = Inception(x,208)#832
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
x = Inception(x,208)
x = Inception(x,256)#1024
x = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x)
x=  Flatten() (x)
x = Dropout(0.4)(x)
x = Dense(100,activation='relu')(x)
x = Dense(21,activation='softmax')(x)
model = Model(inpt,x,name='inception')
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.summary()
train=model.fit(X_train,Y_train,epochs=15,validation_split=0.1, callbacks=[tensorboard])
model.save("modelgoo.h5")
import tensorflow as tf
sess=tf.Session()
writer = tf.summary.FileWriter("logs",sess.graph)
