#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def k_mean_init_centroids(x, k):
    centroids = np.zeros((k, x.shape[1]))
    randidx = np.random.permutation(x.shape[0])
    centroids = x[randidx[0:k], :]
    return centroids
