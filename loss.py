from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def forward(self, preds, loc_targets, conf_targets):
        '''
        Args:
          preds: (tensor) model outputs, sized [batch_size,150,fmsize,fmsize].
          loc_targets: (tensor) loc targets, sized [batch_size,5,4,fmsize,fmsize].
          conf_targets: (tensor) conf targets, sized [batch_size,5,21,fmsize,fmsize].

        Returns:
          (tensor) loss = 5*SmoothL1Loss(loc) + CrossEntropyLoss(pos_conf) + 0.5*CrossEntropyLoss(neg_conf)
        '''
        batch_size, _, fmsize, _ = preds.size()

        preds = preds.view(batch_size, 5, 4+21, fmsize, fmsize)
        xy = preds[:,:,:2,:,:].sigmoid()  # x->sigmoid(x), y->sigmoid(y)
        wh = preds[:,:,2:4,:,:]
        loc = torch.cat([xy,wh], 2)  # [N,5,4,13,13]

        pos = conf_targets > 0  # [N,13,13]
        num_pos = pos.data.long().sum()
        mask = pos[:,None,None,:,:].expand_as(loc)  # [N,13,13] -> [N,1,1,13,13] -> [N,5,4,13,13]
        loc_loss = F.smooth_l1_loss(loc[mask], loc_targets[mask], size_average=False)

        conf = preds[:,:,4:,:,:]  # [N,5,21,13,13]
        conf = conf.permute(0,1,3,4,2).contiguous().view(-1,21)  # [N,5,21,13,13] -> [N*5*13*13,21]
        conf_targets = conf_targets.unsqueeze(1).expand(batch_size,5,fmsize,fmsize).contiguous().view(-1)  # [N,13,13] -> [N,1,13,13] -> [N*5*13*13,]

        pos_mask = pos.unsqueeze(1).expand(batch_size,5,fmsize,fmsize).contiguous().view(-1)  # [N,13,13] -> N[N,1,13,13] -> [N*5*13*13,]
        neg_mask = 1 - pos_mask
        pos_conf_loss = F.cross_entropy(conf[pos_mask.data.nonzero().squeeze(1)], conf_targets[pos_mask])
        neg_conf_loss = F.cross_entropy(conf[neg_mask.data.nonzero().squeeze(1)], conf_targets[neg_mask])
        conf_loss = pos_conf_loss + 0.5*neg_conf_loss
        print('%f %f %f' % (5*loc_loss.data[0]/(5*num_pos), pos_conf_loss.data[0], 0.5*neg_conf_loss.data[0]), end=' ')
        return 5*loc_loss / (5*num_pos) + conf_loss
