import body_net
import cv2
import torch

if __name__ == '__main__':
    extractor, classifier = body_net.vgg16_decompose()
    rpn = body_net.RegionProposalNetwork()
    head = body_net.VGG16RoIHead(classifier)
        
    submod = body_net.FasterRCNNVGG16(extractor, rpn, head)
    net = body_net.FasterRCNNTrainer(submod).cuda()

    print("loading model...")
    net.load_state_dict(torch.load('fasterRCNN.pth'))
    print("successfully load faster rcnn.")

    net.eval()
    #print(net)

    img = cv2.imread("imgs/1.jpg")
    bbox = net.get_all_locs(img)

    img = body_net.print_rectangle(img, bbox)
    cv2.imwrite("det/ans.jpg", img)
    cv2.imshow("faster rcnn detection", img)
    cv2.waitKey(0)
    exit(0)