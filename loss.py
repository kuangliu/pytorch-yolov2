from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()

    def forward(self, preds, loc_targets, conf_targets, prob_targets):
        '''
        Args:
          preds: (tensor) model outputs, sized [batch_size,150,fmsize,fmsize].
          loc_targets: (tensor) loc targets, sized [batch_size,5,4,fmsize,fmsize].
          conf_targets: (tensor) conf targets, sized [batch_size,5,fmsize,fmsize].
          prob_targets: (tensor) prob targets, sized [batch_size,5,fmsize,fmsize].

        Returns:
          (tensor) loss = SmoothL1Loss(loc) + CrossEntropyLoss(conf) + SmoothL1Loss(prob)
        '''
        batch_size, _, fmsize, _ = preds.size()
        preds = preds.view(batch_size, 5, 4+1+20, fmsize, fmsize)

        ### loc_loss
        xy = preds[:,:,:2,:,:].sigmoid()  # x->sigmoid(x), y->sigmoid(y)
        wh = preds[:,:,2:4,:,:].exp()
        loc = torch.cat([xy,wh], 2)  # [N,5,4,13,13]

        pos = conf_targets > 0  # [N,5,13,13]
        num_pos = pos.data.long().sum()

        mask = pos.unsqueeze(2).expand_as(loc)  # [N,5,13,13] -> [N,5,1,13,13] -> [N,5,4,13,13]
        loc_loss = F.smooth_l1_loss(loc[mask], loc_targets[mask], size_average=False)

        ### prob_loss
        prob = preds[:,:,4,:,:].sigmoid()  # [N,5,13,13]
        neg = 1 - pos
        prob_loss = F.smooth_l1_loss(prob[pos], prob_targets[pos], size_average=False) \
                  + F.smooth_l1_loss(prob[neg], prob_targets[neg], size_average=False) * 0.5

        ### conf_loss
        conf = preds[:,:,5:,:,:]  # [N,5,20,13,13]
        conf = conf * prob.unsqueeze(2).expand_as(conf)
        conf = conf.permute(0,1,3,4,2).contiguous().view(-1,20)  # [N,5,21,13,13] -> [N*5*13*13,21]
        pos_mask = pos.view(-1)  # [N,5,13,13] -> [N*5*13*13,]
        conf_loss = F.cross_entropy(conf[pos_mask.data.nonzero().squeeze(1)], conf_targets[pos_mask]-1, size_average=False)

        print('%f %f %f' % (loc_loss.data[0]/num_pos, prob_loss.data[0]/num_pos, conf_loss.data[0]/num_pos), end=' ')
        return loc_loss/num_pos + prob_loss/num_pos + conf_loss/num_pos
