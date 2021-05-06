#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/05/06
"""
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision.transforms import transforms
import numpy as np


class GarbageDataset(Dataset):
    def __init__(self, data_root, mode='train', train_test_list=''):
        super(GarbageDataset, self).__init__()
        self.mode = mode
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
            transforms.Normalize([0.68768471, 0.58804662, 0.49177842], [0.18491538, 0.23386126, 0.28080707])
        ]) if mode == 'train' else transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize([0.68768471, 0.58804662, 0.49177842], [0.18491538, 0.23386126, 0.28080707])
        ])
        self.prepare_data()

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_root, self.train_test_list)):
            print("train or test list file not exist")
        with open(os.path.join(self.data_root, self.train_test_list), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                image_path = str(line[0])
                image = os.path.join(self.data_root, image_path)
                label = int(line[1])
                self.images.append(image)
                self.labels.append(label)

    def __getitem__(self, index):
        image = self.images[index]
        im = Image.open(image)
        print(im.size)
        w, h = im.size
        l = max(w, h)
        dx, dy = 0, 0
        if w > h:
            dy = (w - h) // 2
        else:
            dx = (h - w) // 2
        im = np.array(im)
        im_temp = np.ones((l, l, 3)).astype(int)
        for i in range(l):
            for j in range(l):
                im_temp[i, j, :] = [127, 127, 127]
        im_temp[dy:dy + h, dx:dx + w, :] = im[:, :, :]
        im = Image.fromarray(np.uint8(im_temp))
        im = self.transforms(im)
        label = self.labels[index]
        return im, label

    def __len__(self):
        return len(self.images)
