import numpy as np
import cv2 as cv
from collections import deque
import baselinemodel as bm
import dataprocess as dp


from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
x , y , input , output = dp.process(X_train,y_train)

model = bm.baseline_model(input,output)
model.fit(x,y)

def predict(k):
    l = np.zeros((1,28,28),dtype=np.uint8)
    l[0]=k

    v = dp.inputProcess(l)

    i = model.predict_class(v)
    print(i)






def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



def start():
    cap = cv.VideoCapture(0)
    lb = np.array([110 ,50  ,50])
    ub = np.array([ 130,255 ,255])
    pts = deque(maxlen = 512)
    blackboard = np.zeros((480,640,3),dtype=np.uint8)
    blackboard  = blackboard
    digit = np.zeros((200,200,3),dtype = np.uint8)
    k12 = np.zeros((28,28),dtype = np.uint8)
    while(cap.isOpened()):
        ret, img = cap.read()
        img = cv.flip(img,1)
        imgHsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        mask = cv.inRange(imgHsv,lb,ub)
        blur = cv.medianBlur(mask,15)
        blur = cv.GaussianBlur(blur,(5,5),0)
        thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        cnts = cv.findContours(thresh.copy(),cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[1]
        center = None
        if len(cnts)>=1:
            contour = max(cnts,key=cv.contourArea)
            if cv.contourArea(contour)>250:
                ((x,y),radius) = cv.minEnclosingCircle(contour)
                cv.circle(img,(int(x),int(y)),int(radius),(0,255,255),2)
                cv.circle(img,center,5,(0,0,255),-1)
                M = cv.moments(contour)
                center = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
                pts.appendleft(center)
                for i in range(1,len(pts)):
                    if pts[i-1] is None or pts[i] is None:
                        continue
                    cv.line(img,pts[i-1],pts[i],(0,0,255),5)
                    cv.line(blackboard,pts[i-1],pts[i],(255,255,255),10)

        elif len(cnts) == 0:

            if len(pts)!=[]:

                blackboard_gray = cv.cvtColor(blackboard,cv.COLOR_BGR2GRAY)
                blur1 = cv.medianBlur(blackboard_gray,15)
                blur1 = cv.GaussianBlur(blur1,(5,5),0)
                thresh1 = cv.threshold(blur1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
                blackboard_cnts = cv.findContours(thresh1.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_NONE)[1]
                if len(blackboard_cnts)>=1:
                    cnt = max(blackboard_cnts,key = cv.contourArea)
                    print(cnt.shape)
                    print(cv.contourArea(cnt))
                    if(cv.contourArea(cnt)>2000):
                        x,y,w,h = cv.boundingRect(cnt)
                        digit = blackboard_gray[y:y+h,x:x+w]
                        digit = cv.bitwise_not(digit)
                        k12 = image_resize(digit,width = 28,height = 28)
                        predict(k12)


                pts = deque(maxlen = 512)
                blackboard = np.zeros((480,480,3),dtype = np.uint8)





        cv.imshow(" ",cv.cvtColor(blackboard,cv.COLOR_BGR2GRAY))

        cv.imshow("d",digit)
        cv.imshow("28",k12)
        print(digit.shape)
        cv.imshow('Frame',img)
        k = cv.waitKey(10)
        if k==27:
            break
    cap.release()
    cv.destroyAllWindows()
start()
