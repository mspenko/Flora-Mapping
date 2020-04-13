import os
import cv2
def draw_rectangle(event,x,y,flags,param):
    global ix, iy
    if event==cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        print("point1:=", x, y)
    elif event==cv2.EVENT_LBUTTONUP:
        print("point2:=", x, y)
        print("width=",x-ix)
        print("height=", y - iy)
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        newimage=img[(iy+5):(y-5),(ix+5):(x-5)]
        cv2.imwrite(os.path.join('/home/jiahui/Desktop/'+"rail.jpg"),newimage)

loc='/home/jiahui/Desktop/Robotics_Lab/movie_file4/nonever/'
location='/home/jiahui/Desktop/Robotics_Lab/movie_file4/data1/'
cade=os.listdir(location)
a=0;
for i in cade:
    img=cv2.imread(os.path.join(location+i))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_rectangle)
    while(1):
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    os.rename(os.path.join('/home/jiahui/Desktop/'+"rail.jpg"), loc+i+'.jpg')  
    cv2.destroyAllWindows()
    


