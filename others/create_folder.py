# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 12:15:38 2017

@author: 91243
"""

import os

path = "F:/pedestrian_detection/46_CUHK Occlusion Dataset/raw_img"
savePath = "F:/pedestrian_detection/46_CUHK Occlusion Dataset/processed_img"

method = 'faster-RCNN'
folderDir = os.listdir(path)

for subFolder in folderDir:
    sub = os.path.join('%s/%s'%(path,subFolder))
    method_path = os.path.join('%s/%s/%s'%(savePath,method,subFolder))
    if not os.path.exists(method_path):
        os.makedirs(method_path)