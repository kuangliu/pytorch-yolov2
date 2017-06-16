'''Encode target locations and class labels.'''
import math
import torch

from utils import iou, meshgrid


class DataEncoder:
    def __init__(self):
        self.anchors = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]  # anchors from Darknet cfg file
        # self.anchors = [(0.6240, 1.2133), (1.4300, 2.2075), (2.2360, 4.3081), (4.3940, 6.5976), (9.5680, 9.9493)]  # anchors I get

    def encode(self, boxes, classes, input_size):
        '''Encode target bounding boxes and class labels into YOLOv2 format.

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax) in range [0,1], sized [#obj, 4].
          classes: (tensor) object class labels, sized [#obj,].
          input_size: (int) model input size.

        Returns:
          (tensor) encoded bounding boxes, sized [5,4,fmsize,fmsize].
          (tensor) class labels, sized [5,fmsize,fmsize].
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

        loc = torch.zeros(5,4,fmsize,fmsize)  # 5boxes * 4coords
        for i, box in enumerate(boxes):
            cx = int(bx[i])
            cy = int(by[i])
            for j, (pw,ph) in enumerate(self.anchors):
                tw = bw[i] / pw
                th = bh[i] / ph
                loc[j,:,cy,cx] = torch.Tensor([tx[i], ty[i], tw, th])

        xy = meshgrid(fmsize, swap_dims=True) + 0.5  # grid center, [fmsize*fmsize,2]
        wh = torch.Tensor(self.anchors)              # [5,2]

        xy = xy.view(fmsize,fmsize,1,2).expand(fmsize,fmsize,5,2)
        wh = wh.view(1,1,5,2).expand(fmsize,fmsize,5,2)
        anchor_boxes = torch.cat([xy-wh/2, xy+wh/2], 3)  # [fmsize,fmsize,5,4]

        ious = iou(anchor_boxes.view(-1,4), boxes/grid_size)  # [fmsize*fmsize*5, N]
        conf = torch.LongTensor(5,fmsize,fmsize).zero_()
        for i in range(num_boxes):
            box_iou = ious[:,i].contiguous().view(fmsize,fmsize,5).permute(2,0,1)  # [5,fmsize,fmsize]
            conf[box_iou>0.5] = 1 + classes[i]
        return loc, conf

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

        conf = outputs[:,4:,:,:]  # [5,21,13,13]
        conf = conf.permute(0,2,3,1).contiguous().view(-1,21)  # [845,21]
        max_conf, max_ids = conf.max(1)  # [845,1]
        ids = max_ids.squeeze(1).nonzero().squeeze(1)  # [#boxes,]
        return boxes[ids] / fmsize


def test():
    boxes = torch.Tensor([[48, 240, 195, 371], [8, 12, 352, 498]])
    classes = torch.LongTensor([11,14])
    w = 353
    h = 500
    boxes[:,0::2] /= w
    boxes[:,1::2] /= h
    input_size = 416

    encoder = DataEncoder()
    loc, conf = encoder.encode(boxes, classes, input_size)
    # print(loc.size())
    # print(conf.size())
    # print(conf)

# test()
