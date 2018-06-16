import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import torchvision
from math import sqrt
from itertools import product
import cv2
import numpy as np

class ssd(nn.Module):
    def __init__(self):
        super(ssd, self).__init__()
        self.vgg = vgg16_convert()
        self.L2Norm = L2Norm(512, 20)
        extras_list = [[1024, 256, (1,1), (1,1), 0],
                      [256, 512, (3,3), (2,2), (1,1)],
                      [512, 128, (1,1), (1,1)],
                      [128, 256, (3,3), (2,2), (1,1)],
                      [256, 128, (1,1), (1,1)],
                      [128, 256, (3,3), (1,1)],
                      [256, 128, (1,1), (1,1)],
                      [128, 256, (3,3), (1,1)]]
        self.extras = down_sample(extras_list)
        loc_list = [[512, 16, (3,3), (1,1), (1,1)],
                    [1024, 24, (3,3), (1,1), (1,1)],
                    [512, 24, (3,3), (1,1), (1,1)],
                    [256, 24, (3,3), (1,1), (1,1)],
                    [256, 16, (3,3), (1,1), (1,1)],
                    [256, 16, (3,3), (1,1), (1,1)]]
        self.loc = down_sample(loc_list)
        conf_list = [[512, 84, (3,3), (1,1), (1,1)],
                     [1024, 126, (3,3), (1,1), (1,1)],
                     [512, 126, (3,3), (1,1), (1,1)],
                     [256, 126, (3,3), (1,1), (1,1)],
                     [256, 84, (3,3), (1,1), (1,1)],
                     [256, 84, (3,3), (1,1), (1,1)]]
        self.conf = down_sample(conf_list)
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(21, 0, 200, 0.01, 0.45)
        self.priorbox = PriorBox()
        self.priors = self.priorbox.forward()

    def forward(self, x):
        x = preprocess(x)
        sources = list()
        loc = list()
        conf = list()

        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k%2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        output = self.detect(
            loc.view(loc.size(0), -1, 4),
            self.softmax(conf.view(conf.size(0), -1, 21)),
            self.priors.type(type(x)))
        return output

    def get_all_locs(self, x):
        with torch.no_grad():
            detections = self(x)
        scale = torch.Tensor(frame.shape[1::-1]).repeat(2)
        i = 15
        j = 0
        bbox = list()
        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            #pt order x1, y1, x2, y2
            bbox.append(pt)
            j+=1

        return bbox

class Detect(Function):
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = [0.1, 0.2]

    def forward(self, loc_data, conf_data, prior_data):
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            for cl in range(1 ,self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                    boxes[ids[:count]]), 1)

        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

class PriorBox(object):
    def __init__(self):
        super(PriorBox, self).__init__()
        self.image_size = 300
        self.num_priors = 6
        self.variance = [0.1, 0.2]
        self.features_maps = [38, 19, 10, 5, 3, 1]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def forward(self):
        mean = []
        for k, f in enumerate(self.features_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size/self.steps[k]
                cx = (j+0.5)/f_k
                cy = (i+0.5)/f_k
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def print_rectangle(img, locs):
    for loc in locs:
        cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 0, 255), 2)
    return img

def preprocess(img):
    img = cv2.resize(img, (300, 300)).astype(np.float32)
    img -= (104.0, 117.0, 123.0)
    img = img.astype(np.float32)
    img = img[:, :, ::-1].copy()
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    return img

def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
    
def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union
        idx = idx[IoU.le(overlap)]
    return keep, count

def vgg16_convert():
    #we don't neet parameters
    model = torchvision.models.vgg16(pretrained=False)  
    # no need for the last maxpooling layer
    extractor = list(model.features)[:-1]
    vgg16 = nn.ModuleList(extractor)
    #change the original fullly connected classifer layer to convolutional layer
    vgg16.append(nn.MaxPool2d(kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1),ceil_mode=False))
    vgg16.append(nn.Conv2d(512, 1024, kernel_size=(3,3),stride=(1,1),padding=(6,6),dilation=(6,6)))
    vgg16.append(nn.ReLU(True))
    vgg16.append(nn.Conv2d(1024,1024,kernel_size=(1,1),stride=(1,1)))
    vgg16.append(nn.ReLU(True))
    return vgg16

def get_conv2d(inp, outp, k, s, p=0):
    return nn.Conv2d(inp, outp, kernel_size=k, stride=s, padding=p)

def down_sample(model_list):
    extractor = nn.ModuleList()
    for ll in model_list:
        extractor.append(get_conv2d(*ll))
    return extractor

if __name__ == '__main__':
    net = ssd().cuda()

    print("loading model...")
    net.load_state_dict(torch.load('ssd300.pth'))
    print("successfully load ssd.")

    net.eval()
    img = cv2.imread('1.jpg')
    img = preprocess(img)

    y = net(img)
    print(y)
