#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import find_closest_centroids as fcc
import compute_centroids as cc


def run_K_mean(x, initial_centroids, max_iters):
    m, n = x.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros((m, 1))
    for i in range(max_iters):
        idx = fcc.find_closest_centroids(x, centroids)
        centroids = cc.compute_centroids(x, idx, K)
    return centroids, idx
