#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def find_closest_centroids(x, centroids):
    m, n = x.shape
    K = centroids.shape[0]
    idx = np.zeros((m, 1))
    for i in range(m):
        dis_s = np.zeros((1, K))
        for k in range(K):
            dis_s[0, k] = np.sum(x[i, :] - centroids[k, :])**2
        idx[i, 0] = np.argmin(dis_s, axis=1)
    return idx
