#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/05/06
"""

import argparse

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from dataset import GarbageDataset
from utils.utils import load_pretrained_weights, build_optimizer, build_scheduler
import torch.nn as nn
import tensorboardX as tb
import torch
import numpy as np


parser = argparse.ArgumentParser(description="garbage classify")
parser.add_argument('--epoch', default=50, type=int, help='train epoch')
parser.add_argument('--batch_size', default=16, type=int, help='train batch size')
args = parser.parse_args()

# epochs = args.epoch
# batch_size = args.batch_size
# pre_trained_weight = args.weight


def train():
    model = resnet50(num_classes=40)
    pre_trained_weight = 'weights/resnet50-19c8e357.pth'
    load_pretrained_weights(model, pre_trained_weight)
    model = model.cuda()
    train_dataset = GarbageDataset(data_root='data/garbage', mode='train', train_test_list='train_list.txt')
    test_dataset = GarbageDataset(data_root='data/garbage', mode='train', train_test_list='test_list.txt')
    loss_func = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optim='adam')
    softmax = nn.Softmax()
    max_epoch = args.epoch
    batch_size = args.batch_size
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=max_epoch)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    cudnn.benchmark = True
    writer = tb.SummaryWriter()
    for epoch in range(max_epoch):
        model.train()
        for index, data in enumerate(train_loader):
            im, label = data
            label = label.cuda()
            im = im.cuda()
            optimizer.zero_grad()
            out = model(im)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            if index % 50 == 0:
                num_epoch = epoch * len(train_loader) + index
                print('{} / {}  loss = {:.6f}'.format(epoch+1, max_epoch, loss.cpu().detach().numpy()[0]))
                writer.add_scalar('loss', loss, num_epoch)

        scheduler.step()
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), 'net_{}.pt'.format(epoch+1))
            model.eval()
            total = len(test_loader)
            sum_correct = 0
            for data in test_loader:
                im, label = data
                im = im.cuda()
                y = model(im)
                y = softmax(y)
                y = y.cpu().detch().numpy()
                idx = np.argmax(y, axis=1)[0]
                sum_correct += idx == label
            accuracy = sum_correct / total
            print('{} / {}  accuracy = {:.6f}'.format(epoch+1, max_epoch, accuracy))
            writer.add_scalar('accuracy', accuracy, epoch+1)

    torch.save(model.state_dict(),  'last.pt')
    writer.close()


if __name__ == '__main__':
    train()
