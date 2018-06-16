# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 10:54:31 2017

@author: 91243
"""

import numpy as np
import cv2
import os
import imutils
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression

path = "F:/pedestrian_detection/46_CUHK Occlusion Dataset/raw_img"
savePath = "F:/pedestrian_detection/46_CUHK Occlusion Dataset/processed_img"
folderDir = os.listdir(path)

#subs = []
#for subFolder in folderDir:
#    subs.append(os.path.join('%s/%s'%(path,subFolder)))
#    
#for sub in subs:
#    allFile = os.listdir(sub)
#    for file in allFile:
#        fileName = os.path.join('%s/%s/%s'%(path,sub,file))
#        #start to process images
#        img = cv2.imread(fileName)
#        hog = cv2.HOGDescriptor()
#        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector)
#        img = imutils.resize(img,width=min(400,img.shape[1]))
#        (rects,weights) = hog.detectMultiScale(img,winStride=(4,4),padding=(8,8),scale=1.05)
#        for(x,y,w,h) in rects:
#            cv2.rectangle(img,(x,y),(x+w,y+h),(0, 0, 255),2)
#        raw_hog_path = os.path.dirname(os.path.dirname(sub))+'/'

for subFolder in folderDir:
    sub = os.path.join('%s/%s'%(path,subFolder))
    allFile = os.listdir(sub)
    for file in allFile:
        fileName = os.path.join('%s/%s/%s'%(path,subFolder,file))
        raw_hog_path = os.path.join('%s/Hog/%s/%s'%(savePath,subFolder,file))
        img = cv2.imread(fileName)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        img = imutils.resize(img,width=min(400,img.shape[1]))
        (rects,weights) = hog.detectMultiScale(img,winStride=(4,4),padding=(8,8),scale=1.05)
        for(x,y,w,h) in rects:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0, 0, 255),2)
        
        if not os.path.exists(os.path.dirname(raw_hog_path)):
            os.makedirs(os.path.dirname(raw_hog_path))
        cv2.imwrite(raw_hog_path,img)
