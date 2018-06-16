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
    f = open("../testall.txt")
    cnt = 0
    imgPath = f.readline()
    while imgPath:
        cnt += 1
        #print(cnt)
        labelPath = imgPath.replace("\n","")
        imgPath = labelPath
        labelPath = labelPath.replace("JPEGImages","labels")
        labelPath = labelPath.replace(".jpg",".txt")
        savePath = labelPath.replace("labels", "rcnnpredict")

        if os.path.getsize(labelPath) == 0:
            imgPath = f.readline()
            continue

        img = cv2.imread(imgPath)
        loc = net.get_all_locs(img)
        
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