from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from utils import box_iou, meshgrid


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def decode_loc(self, loc_preds):
        '''Recover predicted locations back to box coordinates.

        Args:
          loc_preds: (tensor) predicted locations, sized [N,5,4,fmsize,fmsize].

        Returns:
          box_preds: (tensor) recovered boxes, sized [N,5,4,fmsize,fmsize].
        '''
        anchors = [(1.3221,1.73145),(3.19275,4.00944),(5.05587,8.09892),(9.47112,4.84053),(11.2364,10.0071)]
        N, _, _, fmsize, _ = loc_preds.size()
        loc_xy = loc_preds[:,:,:2,:,:]   # [N,5,2,13,13]
        grid_xy = meshgrid(fmsize, swap_dims=True).view(fmsize,fmsize,2).permute(2,0,1)  # [2,13,13]
        grid_xy = Variable(grid_xy.cuda())
        box_xy = loc_xy.sigmoid() + grid_xy.expand_as(loc_xy)  # [N,5,2,13,13]

        loc_wh = loc_preds[:,:,2:4,:,:]  # [N,5,2,13,13]
        anchor_wh = torch.Tensor(anchors).view(1,5,2,1,1).expand_as(loc_wh)  # [N,5,2,13,13]
        anchor_wh = Variable(anchor_wh.cuda())
        box_wh = anchor_wh * loc_wh.exp()  # [N,5,2,13,13]
        box_preds = torch.cat([box_xy-box_wh/2, box_xy+box_wh/2], 2)  # [N,5,4,13,13]
        return box_preds

    def forward(self, preds, loc_targets, cls_targets, box_targets):
        '''
        Args:
          preds: (tensor) model outputs, sized [batch_size,150,fmsize,fmsize].
          loc_targets: (tensor) loc targets, sized [batch_size,5,4,fmsize,fmsize].
          cls_targets: (tensor) conf targets, sized [batch_size,5,20,fmsize,fmsize].
          box_targets: (list) box targets, each sized [#obj,4].

        Returns:
          (tensor) loss = SmoothL1Loss(loc) + SmoothL1Loss(iou) + SmoothL1Loss(cls)
        '''
        batch_size, _, fmsize, _ = preds.size()
        preds = preds.view(batch_size, 5, 4+1+20, fmsize, fmsize)

        ### loc_loss
        xy = preds[:,:,:2,:,:].sigmoid()   # x->sigmoid(x), y->sigmoid(y)
        wh = preds[:,:,2:4,:,:].exp()
        loc_preds = torch.cat([xy,wh], 2)  # [N,5,4,13,13]

        pos = cls_targets.max(2)[0].squeeze() > 0  # [N,5,13,13]
        num_pos = pos.data.long().sum()
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,5,13,13] -> [N,5,1,13,13] -> [N,5,4,13,13]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=False)

        ### iou_loss
        iou_preds = preds[:,:,4,:,:].sigmoid()  # [N,5,13,13]
        iou_targets = Variable(torch.zeros(iou_preds.size()).cuda()) # [N,5,13,13]
        box_preds = self.decode_loc(preds[:,:,:4,:,:])  # [N,5,4,13,13]
        box_preds = box_preds.permute(0,1,3,4,2).contiguous().view(batch_size,-1,4)  # [N,5*13*13,4]
        for i in range(batch_size):
            box_pred = box_preds[i]  # [5*13*13,4]
            box_target = box_targets[i]  # [#obj, 4]
            iou_target = box_iou(box_pred, box_target)  # [5*13*13, #obj]
            iou_targets[i] = iou_target.max(1)[0].view(5,fmsize,fmsize)  # [5,13,13]

        mask = Variable(torch.ones(iou_preds.size()).cuda()) * 0.1  # [N,5,13,13]
        mask[pos] = 1
        iou_loss = F.smooth_l1_loss(iou_preds*mask, iou_targets*mask, size_average=False)

        ### cls_loss
        cls_preds = preds[:,:,5:,:,:]  # [N,5,20,13,13]
        cls_preds = cls_preds.permute(0,1,3,4,2).contiguous().view(-1,20)  # [N,5,20,13,13] -> [N,5,13,13,20] -> [N*5*13*13,20]
        cls_preds = F.softmax(cls_preds)  # [N*5*13*13,20]
        cls_preds = cls_preds.view(batch_size,5,fmsize,fmsize,20).permute(0,1,4,2,3)  # [N*5*13*13,20] -> [N,5,20,13,13]
        pos = cls_targets > 0
        cls_loss = F.smooth_l1_loss(cls_preds[pos], cls_targets[pos], size_average=False)

        print('%f %f %f' % (loc_loss.data[0]/num_pos, iou_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' ')
        return (loc_loss + iou_loss + cls_loss) / num_pos
