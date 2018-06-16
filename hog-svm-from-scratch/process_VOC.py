import numpy as np
import cv2
import os
import imutils
from imutils.object_detection import non_max_suppression
from ipdb import set_trace
import time

f = open("testall.txt")
cnt = 0
imgPath = f.readline()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

start = time.time()
while imgPath:
    cnt += 1
    print(cnt)
    if(cnt in [1024,5328,9797]):
        #unknown error, just simply skip it
        imgPath = f.readline()
        continue
    labelPath = imgPath.replace("\n","")
    imgPath = labelPath
    labelPath = labelPath.replace("JPEGImages","labels")
    labelPath = labelPath.replace(".jpg",".txt")
    savePath = labelPath.replace("labels", "HoGpredict")
    if os.path.getsize(labelPath) == 0:
        imgPath = f.readline()
        continue

    img = cv2.imread(imgPath)
    img = imutils.resize(img,width=min(300,img.shape[1]))
    (rects,weights) = hog.detectMultiScale(img,winStride=(4,4),padding=(8,8),scale=1.05)

    loc = []

    for (x, y, w, h) in rects:
        loc.append([x, x+w, y, y+h])

    outfile = open(savePath, "w")
    for i in range(len(loc)):
        outlines = '0'
        for j in range(4):
            outlines = outlines + ' ' + str(loc[i][j])
        outlines = outlines + '\n'
        #set_trace()
        outfile.write(outlines)
    outfile.close()
    imgPath = f.readline()

f.close()
total_time = time.time() - start
print("total_time=",total_time)
print("total_img=",cnt)
print("avg speed=",total_time/cnt)