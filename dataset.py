#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/05/06
"""

from torch.utils.data import Dataset
import os
from torchvision.transforms import transforms


class GarbageDataset(Dataset):
    def __init__(self, data_root, mode='train', image_path='', train_test_list=''):
        super(GarbageDataset, self).__init__()
        self.mode = mode
        self.image_path = image_path
        self.train_test_list = train_test_list
        self.data_root = data_root
        self.images = []
        self.labels = []
        self.transforms = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([], [])
        ]) if mode == 'train' else transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize([], [])
        ])

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_root, self.train_test_list)):
            print("train or test list file not exist")
        with open(os.path.join(self.data_root, self.train_test_list), 'rb') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                image = os.path.join(self.data_root, line[0])
                label = int(line[1])
                self.images.append(image)
                self.labels.append(label)

    def __getitem__(self, index):
        r
