'''Encode target locations and class labels.'''
import math
import torch

from utils import iou, nms, meshgrid, softmax


class DataEncoder:
    def __init__(self):
        self.anchors = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]  # anchors from Darknet cfg file
        #self.anchors = [(0.6240, 1.2133), (1.4300, 2.2075), (2.2360, 4.3081), (4.3940, 6.5976), (9.5680, 9.9493)]  # anchors I get

    def encode(self, boxes, labels, input_size):
        '''Encode target bounding boxes and class labels into YOLOv2 format.

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax) in range [0,1], sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int) model input size.

        Returns:
          loc: (tensor) encoded bounding boxes, sized [5,4,fmsize,fmsize].
          conf: (tensor) class labels, sized [5,fmsize,fmsize].
          prob: (tensor) probability of cell is responsible for predicting the object, sized [5,fmsize,fmsize].
        '''
        num_boxes = len(boxes)
        # input_size -> fmsize
        # 320->10, 352->11, 384->12, 416->13, ..., 608->19
        fmsize = (input_size - 320) / 32 + 10
        grid_size = input_size / fmsize

        boxes *= input_size  # scale [0,1] -> [0,input_size]
        bx = (boxes[:,0] + boxes[:,2]) * 0.5 / grid_size  # in [0,fmsize]
        by = (boxes[:,1] + boxes[:,3]) * 0.5 / grid_size  # in [0,fmsize]
        bw = (boxes[:,2] - boxes[:,0]) / grid_size        # in [0,fmsize]
        bh = (boxes[:,3] - boxes[:,1]) / grid_size        # in [0,fmsize]

        tx = bx - bx.floor()
        ty = by - by.floor()

        xy = meshgrid(fmsize, swap_dims=True) + 0.5  # grid center, [fmsize*fmsize,2]
        wh = torch.Tensor(self.anchors)              # [5,2]

        xy = xy.view(fmsize,fmsize,1,2).expand(fmsize,fmsize,5,2)
        wh = wh.view(1,1,5,2).expand(fmsize,fmsize,5,2)
        anchor_boxes = torch.cat([xy-wh/2, xy+wh/2], 3)  # [fmsize,fmsize,5,4]

        ious = iou(anchor_boxes.view(-1,4), boxes/grid_size)  # [fmsize*fmsize*5, N]
        ious = ious.view(fmsize,fmsize,5,num_boxes)           # [fmsize,fmsize,5,N]
        prob = ious.max(3)[0].squeeze().permute(2,0,1)

        loc = torch.zeros(5,4,fmsize,fmsize)  # 5boxes * 4coords
        conf = torch.LongTensor(5,fmsize,fmsize).zero_()
        for i in range(num_boxes):
            cx = int(bx[i])
            cy = int(by[i])
            _, max_idx = ious[cy,cx,:,i].max(0)
            j = max_idx[0]
            conf[:,cy,cx][j] = labels[i] + 1

            tw = bw[i] / self.anchors[j][0]
            th = bh[i] / self.anchors[j][1]
            loc[j,:,cy,cx] = torch.Tensor([tx[i], ty[i], tw, th])
        return loc, conf, prob

    def decode(self, outputs, input_size):
        '''Transform predicted loc/conf back to real bbox locations and class labels.

        Args:
          outputs: (tensor) model outputs, sized [1,125,13,13].
          input_size: (int) model input size.

        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].
        '''
        fmsize = outputs.size(2)
        outputs = outputs.view(5,25,13,13)

        loc_xy = outputs[:,:2,:,:]   # [5,2,13,13]
        grid_xy = meshgrid(fmsize, swap_dims=True).view(fmsize,fmsize,2).permute(2,0,1)  # [2,13,13]
        box_xy = loc_xy.sigmoid() + grid_xy.expand_as(loc_xy)  # [5,2,13,13]

        loc_wh = outputs[:,2:4,:,:]  # [5,2,13,13]
        anchor_wh = torch.Tensor(self.anchors).view(5,2,1,1).expand_as(loc_wh)  # [5,2,13,13]
        box_wh = anchor_wh * loc_wh.exp()  # [5,2,13,13]

        boxes = torch.cat([box_xy-box_wh/2, box_xy+box_wh/2], 1)  # [5,4,13,13]
        boxes = boxes.permute(0,2,3,1).contiguous().view(-1,4)    # [845,4]

        prob = outputs[:,4,:,:].sigmoid()  # [5,13,13]
        conf = outputs[:,5:,:,:]  # [5,20,13,13]
        conf = conf.permute(0,2,3,1).contiguous().view(-1,20)
        conf = softmax(conf)  # [5*13*13,20]

        score = conf * prob.view(-1).unsqueeze(1).expand_as(conf)  # [5*13*13,20]
        score = score.max(1)[0].view(-1)  # [5*13*13,]
        ids = (score>0.5).nonzero().squeeze()
        keep = nms(boxes[ids], score[ids])
        return boxes[ids][keep] / fmsize


def test():
    boxes = torch.Tensor([[48, 240, 195, 371], [8, 12, 352, 498]])
    labels = torch.LongTensor([11,14])
    w = 353
    h = 500
    boxes[:,0::2] /= w
    boxes[:,1::2] /= h
    input_size = 416

    encoder = DataEncoder()
    loc, conf = encoder.encode(boxes, labels, input_size)
    # print(loc.size())
    # print(conf.size())
    # print(conf)

# test()
