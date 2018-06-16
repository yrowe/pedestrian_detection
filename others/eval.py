import numpy as np
from ipdb import set_trace
import os
import pandas as pd

#when calculate ssd or yolo, just change the corresponding path.

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

def compute_predict():
    predictPath = "F:/PASCAL_VOC_LINUX/PASCAL_VOC/VOCdevkit/VOC2007/rcnnpredict"
    labelPath = "F:\PASCAL_VOC\VOCdevkit\VOC2007\labels"
    savePath = "F:/PASCAL_VOC_LINUX/PASCAL_VOC/VOCdevkit/VOC2007/rcnnIOU"

    precisionPath = "F:/graduation_project/rcnn_precision"
    recallPath = "F:/graduation_project/rcnn_recall"

    totalFile = os.listdir(predictPath)
    for fileName in totalFile:
        file = predictPath + '/' +fileName
        gtfile = labelPath + '/' + fileName
        if os.path.getsize(file) == 0:
            #maybe when calculate recall, it shouldnot be skipped.
            #outp = open(savePath+'/'+fileName, "w")
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

        mat = np.ones((len1, len2))

        for i in range(len1):
            for j in range(len2):
                iou = bbox_iou(bbox1[i], bbox2[j])
                mat[i][j] = iou
        df = pd.DataFrame(mat)
        threshList = [0.2,0.3,0.4,0.5]
        for thresh in threshList:
            df = pd.DataFrame(mat)
            totalPredictNum = df.index.shape[0]
            totalgtNum = df.columns.shape[0]
            truePredictNum = 0
            for i in range(df.columns.shape[0]):
                tmpIndex = df[i].argmax()
                if df[i][tmpIndex] > thresh:
                    truePredictNum += 1

                df = df.drop(tmpIndex)
                if df.index.shape[0] == 0:
                    break
            precision = truePredictNum/totalPredictNum
            recall = truePredictNum/totalgtNum

            #add mode is "write".
            with open(precisionPath+"{}.txt".format(thresh), "a+") as fp:
                fp.write(str(precision)+'\n')

            with open(recallPath+"{}.txt".format(thresh), "a+") as fp:
                fp.write(str(recall)+'\n')

def compute_locate():
    predictPath = "F:/PASCAL_VOC_LINUX/PASCAL_VOC/VOCdevkit/VOC2007/rcnnpredict"
    labelPath = "F:\PASCAL_VOC\VOCdevkit\VOC2007\labels"
    savePath = "F:/PASCAL_VOC_LINUX/PASCAL_VOC/VOCdevkit/VOC2007/rcnnIOU"

    locatePath = "F:/graduation_project/rcnn_locate.txt"

    locatingDis = 0  #total distance for the whole dataset.
    num = 0  # num of valid detections. 
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

        #ans = ""
        mat = np.ones((len1, len2))

        for i in range(len1):
            for j in range(len2):
                iou = bbox_iou(bbox1[i], bbox2[j])
                #ans += str(iou) + ' '
                mat[i][j] = iou
            #ans += '\n'
        df = pd.DataFrame(mat)

        for i in range(len2):
            tmpIndex = df[i].argmax()
            if df[i][tmpIndex] > 0.2:
                locatingDis += np.abs(bbox1[tmpIndex] - bbox2[i]).sum()
                num += 1

        with open(locatePath, "w") as fp:
            fp.write(str(locatingDis/num))


if __name__ == '__main__':
    compute_predict()
    compute_locate()
    