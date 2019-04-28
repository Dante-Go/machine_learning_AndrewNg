#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def compute_centroids(x, idx, k):
    m, n = x.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        centroids[i, :] = np.sum(x[(idx==i).flatten(), :], axis=0) / np.size(x[(idx==i).flatten(), :], 0)
    return centroids
