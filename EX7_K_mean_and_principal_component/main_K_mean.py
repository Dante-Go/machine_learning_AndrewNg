#!/usr/bin/python3
# -*- coding:utf-8 -*-
import scipy.io as sio
from scipy.misc import imread, imshow, imsave
import numpy as np
import find_closest_centroids as fcc
import compute_centroids as cc
import run_K_means as rkm
import k_mean_init_centroids as kmic
import matplotlib.pyplot as plt

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
    initial_centroids_img = kmic.k_mean_init_centroids(X_img, K_img)
    print(initial_centroids_img.shape)
    centroids_img, idx_img = rkm.run_K_mean(X_img, initial_centroids_img, max_iters)
    X_recovered = np.zeros(X_img.shape)
    for i in range(centroids_img.shape[0]):
        X_recovered[(idx_img==i).flatten(), :] = centroids_img[i, :]
    m_a, n_a, z_a = A.shape
    X_recovered_reshape = np.reshape(X_recovered, (m_a, n_a, z_a))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(A)
    axes[1].imshow(X_recovered_reshape)


