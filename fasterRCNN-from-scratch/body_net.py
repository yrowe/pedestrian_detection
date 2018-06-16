import torch
import torch.nn as nn
import torchvision.models
from ipdb import set_trace
import numpy as np
import cv2
from torch.nn import functional as F
import time
import pandas as pd
from skimage import transform as sktsf


class FasterRCNNTrainer(nn.Module):
    def __init__(self, fasterRCNN):
        super(FasterRCNNTrainer, self).__init__()
        self.faster_rcnn = fasterRCNN
        #generate anchor_base?
        self.anchor_base = np.array([[ -37.254833,  -82.50967 ,   53.254833,   98.50967 ],
                                     [ -82.50967 , -173.01933 ,   98.50967 ,  189.01933 ],
                                     [-173.01933 , -354.03867 ,  189.01933 ,  370.03867 ],
                                     [ -56.      ,  -56.      ,   72.      ,   72.      ],
                                     [-120.      , -120.      ,  136.      ,  136.      ],
                                     [-248.      , -248.      ,  264.      ,  264.      ],
                                     [ -82.50967 ,  -37.254833,   98.50967 ,   53.254833],
                                     [-173.01933 ,  -82.50967 ,  189.01933 ,   98.50967 ],
                                     [-354.03867 , -173.01933 ,  370.03867 ,  189.01933 ]],
                                     dtype=np.float32)
        self.feat_stride = 16
        self.spatial_pooling = torch.nn.AdaptiveMaxPool2d((7, 7))
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
        self.nms_thresh = 0.3
        self.score_thresh = 0.7
        self.n_class = 21

    def forward(self, x, scale):
        #extract feature network, reuse of vgg16.
        #input format = [B, C, H, W]
        img_size = (x.shape[2], x.shape[3])
        x = self.faster_rcnn.extractor(x)
        #now we got feature map
        h = x.shape[2]
        w = x.shape[3]

        anchor = generate_anchors(self.anchor_base, 
                   self.feat_stride, h, w)
        #set_trace()
        n_anchor = self.anchor_base.shape[0]    
        #one more 3*3 conv to extractor features.
        layer1 = F.relu(self.faster_rcnn.rpn.conv1(x))
        #now we need to forward into 2 paths.
        #location path:
        rpn_locs = self.faster_rcnn.rpn.loc(layer1)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        
        #score path:
        rpn_scores = self.faster_rcnn.rpn.score(layer1)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_fg_scores = rpn_scores.view(1, h, w, n_anchor, 2)[:, :, :, :, 1].contiguous().view(1, -1)
        rpn_scores = rpn_scores.view(1, -1, 2)

        rois = self.proposal_layer(
                rpn_locs[0].cpu().numpy(),
                rpn_fg_scores[0].cpu().numpy(),
                anchor, img_size, scale=scale)

        pool = torch.Tensor().cuda()
        for i in range(rois.shape[0]):
            tmp = (rois[i]/self.feat_stride).astype("int")
            inp = x[:, :, tmp[0]:(tmp[2]+1), tmp[1]:(tmp[3]+1)]
            outp = self.spatial_pooling(inp)
            pool = torch.cat((pool, outp))
        #we suppose got a 300*512*7*7 tensor
        pool = pool.view(pool.size(0), -1)

        fc7 = self.faster_rcnn.head.classifier(pool)
        
        roi_cls_locs = self.faster_rcnn.head.cls_loc(fc7)
        roi_scores = self.faster_rcnn.head.score(fc7)

        return roi_cls_locs, roi_scores, rois

    def proposal_layer(self, loc, score, anchor, img_size, scale):
        nms_thresh = 0.7
        pre_nms = 2000      #12000
        post_nms = 300       #2000
        min_size = 16
        roi = loc2bbox(anchor, loc)

        roi[:, [0, 2]] = np.clip(roi[:, [0, 2]], 0, img_size[0])
        roi[:, [1, 3]] = np.clip(roi[:, [1, 3]], 0, img_size[1])
        min_size = min_size*scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]

        keep = np.where((hs >= min_size)&(ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]
        order = score.argsort()[::-1]
        order = order[:pre_nms]
        roi = roi[order, :]
        keep = non_maximum_suppress(roi, nms_thresh)
        keep = keep[:post_nms]
        roi = roi[keep]

        return roi

    def suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()

        for l in range(1, self.n_class):
            if l != 15:
                #if it were not for the person item, just continue. class 0 refers to background.
                continue
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppress(cls_bbox_l, self.nms_thresh, prob_l)

            order = prob_l.argsort()[::-1].astype(np.int32)
            prob_l = prob_l[order]
            cls_bbox_l = cls_bbox_l[order]

            bbox.append(cls_bbox_l[keep])
            label.append((l-1)*np.ones((len(keep), )))
            score.append(prob_l[keep])

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        return bbox, label, score

    def get_all_locs(self, img):
        size = img.shape[0:2]
        img, scale = preprocess(img)
        img = torch.from_numpy(img).cuda().float().unsqueeze(0)
        with torch.no_grad():
            roi_cls_loc, roi_scores, roi = self(img, scale=scale)

        roi = torch.from_numpy(roi).cuda()/scale
        mean = torch.Tensor(self.loc_normalize_mean).cuda().repeat(self.n_class)[None]
        std = torch.Tensor(self.loc_normalize_std).cuda().repeat(self.n_class)[None]

        roi_cls_loc = (roi_cls_loc*std + mean)
        roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)

        roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
        cls_bbox = loc2bbox(roi.cpu().numpy().reshape((-1, 4)), roi_cls_loc.cpu().numpy().reshape((-1, 4)))
        cls_bbox = torch.from_numpy(cls_bbox).cuda()
        cls_bbox = cls_bbox.view(-1, self.n_class*4)

        cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
        cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

        prob = F.softmax(roi_scores, dim=1)

        cls_bbox = cls_bbox.cpu().numpy()
        prob = prob.cpu().numpy()

        bbox, label, score = self.suppress(cls_bbox, prob)
        bbox = bbox[:, [1, 0, 3, 2]]
        bbox1 = list()
        for k, i in enumerate(label):
            bb = [int(b) for b in bbox[k]]
            bbox1.append(bb)

        return bbox1


class FasterRCNNVGG16(nn.Module):
    def __init__(self, extractor, rpn, head):
        super(FasterRCNNVGG16, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head


class RegionProposalNetwork(nn.Module):
    def __init__(self):
        super(RegionProposalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1),padding=(1,1))
        self.score = nn.Conv2d(512, 18, kernel_size=(1,1), stride=(1,1))
        self.loc = nn.Conv2d(512, 36, kernel_size=(1,1), stride=(1,1))

class VGG16RoIHead(nn.Module):
    def __init__(self, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        self.cls_loc = nn.Linear(in_features=4096, out_features=84, bias=True)
        self.score = nn.Linear(in_features=4096, out_features=21, bias= True)


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def non_maximum_suppress(roi, nms_thresh, score=None):
    roi_size = roi.shape[0]
    if score is not None:
        order = score.argsort()[::-1].astype(np.int32)
        roi = roi[order, :]
    roi = torch.from_numpy(roi)
    discard_index = []
    for i in range(roi_size):
        if i in discard_index:
            continue
        try:
            ious = bbox_iou(roi[i].unsqueeze(0), roi[i+1:])
        except ValueError:
            break
        except IndexError:
            break

        tmp_index = np.where(ious > nms_thresh)[0]
        for k in tmp_index:
            discard_index.append(k+i+1)

    all_index = range(roi_size)
    keep = list(set(all_index).difference(set(discard_index)))
    #set_trace()
    keep.sort()

    return keep

def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    p_h = src_bbox[:, 2] - src_bbox[:, 0]
    p_w = src_bbox[:, 3] - src_bbox[:, 1]
    p_y = src_bbox[:, 0] + 0.5 * p_h
    p_x = src_bbox[:, 1] + 0.5 * p_w

    t_y = loc[:, 0]
    t_x = loc[:, 1]
    t_h = loc[:, 2]
    t_w = loc[:, 3]

    ctr_y = t_y * p_h + p_y
    ctr_x = t_x * p_w + p_x

    h = np.exp(t_h) * p_h
    w = np.exp(t_w) * p_w

    roi = np.zeros(loc.shape, dtype=loc.dtype)

    roi[:, 0:1] = (ctr_y - 0.5*h)[:, np.newaxis]
    roi[:, 1:2] = (ctr_x - 0.5*w)[:, np.newaxis]
    roi[:, 2:3] = (ctr_y + 0.5*h)[:, np.newaxis]
    roi[:, 3:4] = (ctr_x + 0.5*w)[:, np.newaxis]

    return roi

def generate_anchors(anchor_base, feat_stride, height, width):
    xx = np.arange(0, width*feat_stride, feat_stride)
    yy = np.arange(0, height*feat_stride, feat_stride)

    shift_x, shift_y = np.meshgrid(xx, yy)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis = 1)

    A = anchor_base.shape[0]
    K = shift.shape[0]

    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1,0,2))
    anchor = anchor.reshape((K*A, 4)).astype(np.float32)

    return anchor


def vgg16_decompose():
    #we don't neet parameters
    model = torchvision.models.vgg16(pretrained=False)  
    # no need for the last maxpooling layer
    extractor = list(model.features)[:-1]
    #no need for the last fulling conncected layer
    classifier = list(model.classifier)[:-1]
    #since we will implement the customed dropout method,
    #we simply discard the original dropout of vgg16
    del classifier[5]
    del classifier[2]
    
    return nn.Sequential(*extractor), nn.Sequential(*classifier)

def preprocess(img, min_size=600, max_size=1000):
    #img is get by func cv2.imread. So its shape is [H, W, C] and its channel format is BGR.
    
    H, W, C = img.shape
    scale1 = min_size/min(H, W)
    scale2 = max_size/max(H, W)
    scale = min(scale1, scale2)
    img = np.transpose(img, (2,0,1))
    
    img = img/255.
    img = sktsf.resize(img, (C, H*scale, W*scale), mode='reflect')

    #return shape should be [C, H ,W]
    #set_trace()
    #img = np.transpose(img, (2,0,1))
    img = img*255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img-mean).astype(np.float32, copy=True)
    return img, scale


def print_rectangle(img, locs):
    for loc in locs:
        cv2.rectangle(img, (loc[0], loc[1]), (loc[2], loc[3]), (0, 0, 255), 2)
    return img