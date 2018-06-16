import body_net
import cv2
import torch


if __name__ == '__main__':
    net = body_net.ssd().cuda()

    print("loading model...")
    net.load_state_dict(torch.load('ssd300.pth'))
    print("successfully load ssd weights.")

    net.eval()
    img = cv2.imread("imgs/1.jpg")
    bbox = net.get_all_locs(img)
    img = body_net.print_rectangle(img, bbox)

    cv2.imwrite("det/ans.jpg", img)
    cv2.imshow("ssd detection", img)
    cv2.waitKey(0)
    exit(0)