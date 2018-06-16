import torch
import cv2
import body_net
import time

if __name__ == '__main__':
    save_path = 'ssd_demo.avi'
    net = body_net.ssd().cuda()

    print("loading model...")
    net.load_state_dict(torch.load('ssd300.pth'))
    print("successfully load ssd weights.")

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Cannot open camera'

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print((frame_width, frame_height))

    #in my GTX 1060, this ssd300 model can reach 23 FPS. set this parameters according to your own machine.
    out = cv2.VideoWriter('{}'.format(save_path), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 23, (frame_width, frame_height))

    start = time.time()
    cnt = 0
    while cap.isOpened():
        cnt += 1
        ret, frame = cap.read()
        if ret:
            bbox = net.get_all_locs(frame)
            frame = body_net.print_rectangle(frame, bbox)
            out.write(frame)
            cv2.imshow("ssd_camera_demo", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        else:
            break

    total_time = time.time() - start
    FPS = round(cnt/total_time, 2)
    print("the average FPS is {}".format(FPS))