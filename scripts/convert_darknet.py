'''Convert pretrained Darknet weights into YOLOv2.

Darknet19 model download from: https://drive.google.com/file/d/0B4pXCfnYmG1WRG52enNpcV80aDg/view
'''
import torch

import numpy as np
import torch.nn as nn

from darknet import Darknet


net = Darknet()
darknet = np.load('./model/darknet19.weights.npz')

# layer1
conv_ids = [0,4,8,11,14,18,21,24,28,31,34,37,40]
for i,conv_id in enumerate(conv_ids):
    net.layer1[conv_id].weight.data = torch.from_numpy(darknet['%d-convolutional/kernel:0' % i].transpose((3,2,0,1)))
    net.layer1[conv_id].bias.data = torch.from_numpy(darknet['%d-convolutional/biases:0' % i])
    bn_id = conv_id + 1
    net.layer1[bn_id].weight.data = torch.from_numpy(darknet['%d-convolutional/gamma:0' % i])
    net.layer1[bn_id].running_mean = torch.from_numpy(darknet['%d-convolutional/moving_mean:0' % i])
    net.layer1[bn_id].running_var = torch.from_numpy(darknet['%d-convolutional/moving_variance:0' % i])

# layer2
conv_ids = [1,4,7,10,13]
for i,conv_id in enumerate(conv_ids):
    net.layer2[conv_id].weight.data = torch.from_numpy(darknet['%d-convolutional/kernel:0' % (13+i)].transpose((3,2,0,1)))
    net.layer2[conv_id].bias.data = torch.from_numpy(darknet['%d-convolutional/biases:0' % (13+i)])
    bn_id = conv_id + 1
    net.layer2[bn_id].weight.data = torch.from_numpy(darknet['%d-convolutional/gamma:0' % (13+i)])
    net.layer2[bn_id].running_mean = torch.from_numpy(darknet['%d-convolutional/moving_mean:0' % (13+i)])
    net.layer2[bn_id].running_var = torch.from_numpy(darknet['%d-convolutional/moving_variance:0' % (13+i)])

torch.save(net.state_dict(), './model/darknet.pth')
