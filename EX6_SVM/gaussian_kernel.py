#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def gaussian_kernel_func(x1, x2, sigma):
    x1 = x1.flatten()
    x2 = x2.flatten()
    sim = np.exp((-1) * np.sum(np.power((x1-x2), 2))/(2*sigma**2))
    return sim
