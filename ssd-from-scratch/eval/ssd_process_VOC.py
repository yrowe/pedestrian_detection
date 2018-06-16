import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.serialization import load_lua
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
from data import VOC_CLASSES as labels
from ssd import build_ssd
from ipdb import set_trace

net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights('../weights/ssd300_mAP_77.43_v2.pth')

f = open("testall.txt")
cnt = 0
imgPath = f.readline()
while imgPath:
    cnt += 1
    print(cnt)
    labelPath = imgPath.replace("\n","")
    imgPath = labelPath
    labelPath = labelPath.replace("JPEGImages","labels")
    labelPath = labelPath.replace(".jpg",".txt")
    savePath = labelPath.replace("labels", "ssdpredict")
    if os.path.getsize(labelPath) == 0:
        imgPath = f.readline()
        continue

    img = cv2.imread(imgPath)
    x = cv2.resize(img, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    detections = y.data
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)
    i = 15
    j = 0
    loc = []
    while detections[0,i,j,0] >= 0.6:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        loc.append([pt[0],pt[0]+pt[2],pt[1],pt[1]+pt[3]])
        j+=1
    
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
