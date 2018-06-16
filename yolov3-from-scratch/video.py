import darknet
import cv2
import time
import os
from ipdb import set_trace
import torch

def camera_detect(save_path = 'camera_deomo.avi'):
    net = darknet.Darknet().cuda()
    print("Loading model...")
    net.module_list.load_state_dict(torch.load('yolov3.pth'))
    print("YOLO has been loaded")

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Cannot open camera'
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print((frame_width,frame_height))

    out = cv2.VideoWriter('{}'.format(save_path), cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))

    start = time.time()
    cnt = 0
    while cap.isOpened():
        cnt += 1
        ret, frame = cap.read()
        if ret:
            loc = net.get_all_predict(frame)
            frame = darknet.print_rectangle(frame, loc)

            out.write(frame)
            cv2.imshow("yolov3_camera_demo", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        else:
            break

    total_time = time.time() - start
    FPS = round(cnt/total_time, 2)
    print("the average FPS is {}".format(FPS))

def video_detect_save(videoPath, save_path = 'outp.avi'):
    net = darknet.Darknet().cuda()
    print("Loading model...")
    net.module_list.load_state_dict(torch.load('yolov3.pth'))
    print("YOLO has been loaded")
    
    cap = cv2.VideoCapture(videoPath)

    assert os.path.exists(videoPath), "the denoted video file does not exist,please check it again."
    assert cap.isOpened(), "faile to open the denoted vido."

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('{}'.format(save_path), cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            loc = net.get_all_predict(frame)
            frame = darknet.print_rectangle(frame, loc)

            out.write(frame)
            cv2.imshow("yolo video detect", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        else:
            break

if __name__ == '__main__':
    #camera_detect()
    video_detect_save('people.mp4')
