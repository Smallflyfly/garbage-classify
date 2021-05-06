#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/05/06
"""

import argparse
from torchvision.models import resnet50

from utils.utils import load_pretrained_weights

parser = argparse.ArgumentParser(description="garbage classify")
parser.add_argument('--epoch', default=50, type=int, help='train epoch')
parser.add_argument('--batch_size', default=16, type=int, help='train batch size')
parser.add_argument('--weight', default='', type=str, help='weight file')
args = parser.parse_args()

epochs = args.epoch
batch_size = args.batch_size
pretrained_weight = args.weight


def train():
    net = resnet50(num_classes=40)
    load_pretrained_weights(net, pretrained_weight)
    dataset =


if __name__ == '__main__':
    train()
