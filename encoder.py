'''Encode target locations and class labels.'''
import math
import torch

import itertools


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
          (tensor) class labels, sized [fmsize,fmsize].
        '''
        num_boxes = len(boxes)
        # input_size -> fmsize
        # 320->10, 352->11, 384->12, 416->13, ..., 608->19
        fmsize = (input_size - 320) / 32 + 10
        grid_size = input_size / fmsize

        boxes *= input_size  # scale [0,1] -> [0,input_size]
        bx = (boxes[:,0] + boxes[:,2]) * 0.5 / grid_size  # [0,fmsize]
        by = (boxes[:,1] + boxes[:,3]) * 0.5 / grid_size  # [0,fmsize]
        bw = (boxes[:,2] - boxes[:,0]) / grid_size        # [0,fmsize]
        bh = (boxes[:,3] - boxes[:,1]) / grid_size        # [0,fmsize]

        tx = (bx - bx.floor()) / fmsize  # [0,1]
        ty = (by - by.floor()) / fmsize  # [0,1]

        loc = torch.zeros(5,4,fmsize,fmsize)  # 5boxes * 4coords
        conf = torch.LongTensor(fmsize,fmsize).zero_()
        for i in range(num_boxes):
            cx = int(bx[i])
            cy = int(by[i])
            conf[cy,cx] = classes[i] + 1
            for j, (pw,ph) in enumerate(self.anchors):
                tw = math.log(bw[i] / pw)
                th = math.log(bh[i] / ph)
                loc[j,:,cy,cx] = torch.Tensor([tx[i], ty[i], tw, th])
        return loc, conf


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
    print(loc[:,7,4])
    print(loc.size())
    print(conf.size())

# test()
