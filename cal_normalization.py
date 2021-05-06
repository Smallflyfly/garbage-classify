#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/12/25
"""

import os
import numpy as np
import cv2

ROOT_PATH = 'data/garbage/train_data/'


def cal_mean_std():
    mean, std = None, None
    images = os.listdir(ROOT_PATH)
    for image in images:
        im = cv2.imread(os.path.join(ROOT_PATH, image))
        im = im[:, :, ::-1] / 255.
        # im = im.reshape(1, im.shape[0], im.shape[1], im.shape[2])
        if mean is None and std is None:
            mean, std = cv2.meanStdDev(im)
        else:
            mean_, std_ = cv2.meanStdDev(im)
            mean_stack = np.stack((mean, mean_), axis=0)
            std_stack = np.stack((std, std_), axis=0)
            mean = np.mean(mean_stack, axis=0)
            std = np.mean(std_stack, axis=0)
    return mean.reshape((1, 3))[0], std.reshape((1, 3))[0]


if __name__ == '__main__':
    mean, std = cal_mean_std()
    print(mean, std)
    # [0.68768471 0.58804662 0.49177842] [0.18491538 0.23386126 0.28080707]
