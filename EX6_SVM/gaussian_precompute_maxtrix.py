#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import gaussian_kernel as gk


def gaussian_matrix(x1, x2, sigma=0.1):
    k = np.zeros((x1.shape[0], x2.shape[0]))
    for i, x_1 in enumerate(x1):
        for j, x_2 in enumerate(x2):
            k[i, j] = gk.gaussian_kernel_func(x_1, x_2, sigma)
    return k
