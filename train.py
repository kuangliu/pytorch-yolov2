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

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), \
           torch.stack([x[1] for x in batch]), \
           torch.stack([x[2] for x in batch]), \
           [x[3] for x in batch]

transform = transforms.Compose([transforms.ToTensor()])

trainset = ListDataset(root='/search/data/user/liukuang/data/VOC2012_trainval_test_images',
                       list_file='./voc_data/voc12_train.txt', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8, collate_fn=collate_fn)

testset = ListDataset(root='/search/data/user/liukuang/data/VOC2012_trainval_test_images',
                      list_file='./voc_data/voc12_test.txt', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8, collate_fn=collate_fn)

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

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()
cudnn.benchmark = True

criterion = YOLOLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (images, loc_targets, cls_targets, box_targets) in enumerate(trainloader):
        images = Variable(images.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        box_targets = [Variable(x.cuda()) for x in box_targets]

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, loc_targets, cls_targets, box_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('%.3f %.3f' % (loss.data[0], train_loss/(batch_idx+1)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (images, loc_targets, cls_targets, box_targets) in enumerate(testloader):
        images = Variable(images.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        box_targets = [Variable(x.cuda()) for x in box_targets]

        outputs = net(images)
        loss = criterion(outputs, loc_targets, cls_targets, box_targets)
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
