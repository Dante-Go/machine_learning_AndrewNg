#!/usr/bin/python3
# -*- coding:utf-8 -*-
import scipy.io as sio
from scipy.misc import imread, imshow, imsave
import numpy as np
import find_closest_centroids as fcc
import compute_centroids as cc
import run_K_means as rkm

if __name__ == '__main__':
    mat = sio.loadmat('ex7data2.mat')
    X = mat['X']
    K = 3
    initial_centroids = np.mat([[3, 3], [6, 2], [8, 5]])
    idx = fcc.find_closest_centroids(X, initial_centroids)
    centroids = cc.compute_centroids(X, idx, K)
    max_iters = 10
    centroids, idx = rkm.run_K_mean(X, initial_centroids, max_iters)
    print(centroids)
    img = imread('bird_small.png')
    print(img.shape)
    A = img / 255
    X_img = np.reshape(A, (img.shape[0]*img.shape[1], 3))
    print(X_img.shape)
    K_img = 16
    initial_centroids_img = 0

