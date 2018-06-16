import torch 
import torch.nn as nn
import cv2 
import darknet
from ipdb import set_trace

if __name__ == '__main__':
    net = darknet.Darknet().cuda()
    net.eval()

    print("Loading model...")
    net.module_list.load_state_dict(torch.load('yolov3.pth'))
    print("YOLO has been loaded")

    img = cv2.imread("imgs/1.jpg")
    loc = net.get_all_predict(img)

    img = darknet.print_rectangle(img, loc)
    save_path = 'demo.jpg'
    
    cv2.imwrite(save_path, img)
    print("save at {}".format(save_path))
    
    #cv2.namedWindow("detection",0)
    #cv2.resizeWindow("detection", 1440, 900);
    
    cv2.imshow("detection",img)
    cv2.waitKey(0)
    exit(0)
    