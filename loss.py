from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def forward(self, preds, loc_targets, conf_targets):
        '''
        Args:
          preds: (tensor) model outputs, sized [batch_size,150,fmsize,fmsize].
          loc_targets: (tensor) loc targets.
          conf_targets: (tensor) conf targets.

        Returns:
          (tensor) loss = SmoothL1Loss(loc) + CrossEntropyLoss(conf)
        '''
        batch_size, _, fmsize, _ = preds.size()

        preds = preds.view(batch_size, 5, 4+21, fmsize, fmsize)
        xy = preds[:,:,:2,:,:].sigmoid()  # x->sigmoid(x), y->sigmoid(y)
        # wh = preds[:,:,2:4,:,:].sqrt()
        wh = preds[:,:,2:4,:,:]
        loc = torch.cat([xy,wh], 2)

        pos = conf_targets > 0
        num_pos = pos.data.long().sum()
        mask = pos.view(batch_size,1,1,fmsize,fmsize).expand_as(loc)
        loc_loss = F.smooth_l1_loss(loc[mask], loc_targets[mask], size_average=False)

        conf = preds[:,:,4:,:,:]  # [N,5,21,13,13]
        conf = conf.permute(0,1,3,4,2).contiguous().view(-1,21)  # [N,5,21,13,13] -> [N,5,13,13,21]
        conf_targets = conf_targets.unsqueeze(1).expand(batch_size,5,fmsize,fmsize)  # [N,13,13] -> [N,1,13,13] -> [N,5,13,13]
        conf_targets = conf_targets.contiguous().view(-1)
        conf_loss = F.cross_entropy(conf, conf_targets, size_average=False)

        print('%f %f' % (loc_loss.data[0]/(5*num_pos), conf_loss.data[0]/(5*num_pos)), end=' ')
        return (loc_loss + conf_loss) / (5*num_pos)
