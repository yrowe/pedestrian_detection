import numpy as np
from ipdb import set_trace
import os
import pandas as pd

def bbox_iou(box1, box2):
    ax1, ax2, ay1, ay2 = box1
    bx1, bx2, by1,by2 = box2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = (inter_x2 - inter_x1)*(inter_y2 - inter_y1)

    a_area = (ax2 - ax1)*(ay2 - ay1)
    b_area = (bx2 - bx1)*(by2 - by1)

    iou = inter_area /(a_area + b_area - inter_area)
    return iou

def compute_iou():
    #predictPath = "F:\PASCAL_VOC\VOCdevkit\VOC2007\yolopredict"
    #labelPath = "F:\PASCAL_VOC\VOCdevkit\VOC2007\labels"
    #savePath = "F:\PASCAL_VOC\VOCdevkit\VOC2007\yoloIOU"

    predictPath = "F:/PASCAL_VOC/VOCdevkit/VOC2007/yolopredict"
    labelPath = "F:/PASCAL_VOC/VOCdevkit/VOC2007/labels"
    savePath = "F:/PASCAL_VOC/VOCdevkit/VOC2007/yoloIOU"

    precisionPath = "./yolo_precision.txt"
    recallPath = "./yolo_recall.txt"

    totalFile = os.listdir(predictPath)

    #yolo predict may be empty
    #but if only yolo predict has some prediction, 
    #there must be corresponding label for ground truth.
    for fileName in totalFile:
        file = predictPath + '/' +fileName
        gtfile = labelPath + '/' + fileName
        if os.path.getsize(file) == 0:
            outp = open(savePath+'/'+fileName,"w")
            continue

        f = open(file)
        line = f.readline()
        bbox1 = []
        while line:
            line = line.replace("\n", "")
            line = line.split(" ")[1:5]
            line = [int(float(i)) for i in line]
            bbox1.append(line)
            line = f.readline()
        f.close()
        bbox1 = np.array(bbox1)
        
        f = open(gtfile)
        line = f.readline()
        bbox2 = []
        while line:
            line = line.replace("\n", "")
            line = line.split(" ")[1:5]
            line = [int(float(i)) for i in line]
            bbox2.append(line)
            line = f.readline()
        f.close()
        bbox2 = np.array(bbox2)

        len1 = len(bbox1)
        len2 = len(bbox2)
        totalGtNum = len2

        #ans = ""
        mat = np.ones((len1, len2))

        for i in range(len1):
            for j in range(len2):
                iou = bbox_iou(bbox1[i], bbox2[j])
                #ans += str(iou) + ' '
                mat[i][j] = iou
            #ans += '\n'
        
        #set_trace()
        thresh_list = [0.2,0.3,0.4,0.5]
        for thresh in thresh_list:
            df = pd.DataFrame(mat)
            totalPredictNum = df.index.shape[0]
            truePredictNum = 0
            for i in range(df.columns.shape[0]):
                tmpIndex = df[i].argmax()
                if df[i][tmpIndex] < thresh:
                     continue
                
                truePredictNum += 1
                df = df.drop(tmpIndex)
                if df.index.shape[0] == 0:
                    break
            
            #set_trace()
            precision = truePredictNum/totalPredictNum 
            recall = truePredictNum/totalGtNum
            
            with open(precisionPath+"-thresh={}".format(thresh),"a+") as fp:
                fp.write(str(precision)+'\n')

            with open(recallPath+"-thresh={}".format(thresh),"a+") as fp:
                fp.write(str(recall)+'\n')
        
        #set_trace()
        #outp = open(savePath+'/'+fileName,"w")
        #outp.write(ans)
        #outp.close()

def cntp_precsion(path = "F:\graduation_project\yolo_precision.txt"):
    df = pd.read_csv("precision.txt", header=None)
    print(df.mean())

if __name__ == '__main__':
    compute_iou()

'''      
a = [[185, 279, 62, 199]]
b = [[177, 288, 63, 175]]

len1 = len(a)
len2 = len(b)

ans = np.ones((len1, len2))

for i in range(len1):
    for j in range(len2):
        iou = bbox_iou(a[i], b[j])
        ans[i][j] = iou

print(ans)
#set_trace()
'''