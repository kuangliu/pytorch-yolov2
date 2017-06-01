from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import YOLOLoss
from darknet import Darknet
from datagen import ListDataset

from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch YOLOv2 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor()])

trainset = ListDataset(root='/search/liukuang/data/VOC2012_trainval_test_images',
                       list_file='./voc_data/voc12_train.txt', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

testset = ListDataset(root='/search/liukuang/data/VOC2012_trainval_test_images',
                      list_file='./voc_data/voc12_test.txt', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

# Model
net = Darknet()
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    # Load pretrained Darknet model
    net.load_state_dict(torch.load('./model/darknet.pth'))

if use_cuda:
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()
    cudnn.benchmark = True

criterion = YOLOLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(trainloader):
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()

        images = Variable(images)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, loc_targets, conf_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('%.3f %.3f' % (loss.data[0], train_loss/(batch_idx+1)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(testloader):
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()

        images = Variable(images, volatile=True)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)

        outputs = net(images)
        loss = criterion(outputs, loc_targets, conf_targets)
        test_loss += loss.data[0]
        print('%.3f %.3f' % (loss.data[0], test_loss/(batch_idx+1)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
