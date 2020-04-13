import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
from keras.models import load_model      

model=load_model('/home/jiahui/modelgoo3.h5')

location='/home/jiahui/Desktop/movie_file2/testdata/tree4/'
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
    gray=np.array(gray.resize((180,180)))
    all_images.append(gray)
    ib.append(i)

shuffle(all_images,ib,random_state=81743)
x_pred=np.array(all_images)/255

cate=os.listdir('/home/jiahui/Desktop/Chinese_Plant_Dataset/insta/')
predind=0;
o=0;
index=0;
while index<len(x_pred):
    img=(np.expand_dims(x_pred[index],0))
    predictions=model.predict(img)
    pred_img=np.argmax(predictions[0])
    if pred_img==predind:o=o+1;     
    index = index+1
    acc=o/100;
    print(pred_img)
print(acc)
