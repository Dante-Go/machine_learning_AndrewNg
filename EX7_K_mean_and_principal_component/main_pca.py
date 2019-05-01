#!/usr/bin/python3
# -*- coding:utf-8 -*-
import scipy.io as sio
import matplotlib.pyplot as plt
import pca_feature_normalize as fn
import pca_run_pca as prp
import pca_project_data as ppd
import pca_recover_data as prd
import pca_display_face_data as pdfd

if __name__ == '__main__':
    mat = sio.loadmat('ex7data1.mat')
    X = mat['X']
    # plt.scatter(X[:, 0], X[:, 1], facecolor='none', edgecolors='b')
    X_norm, mu, sigma = fn.feature_normalize(X)
    U, S, V = prp.run_pca(X_norm)
    K = 1
    Z = ppd.project_data(X_norm, U, K)
    X_rec = prd.recover_data(Z, U, K)
    plt.scatter(X_norm[:, 0], X_norm[:, 1], s=30, facecolors='none', edgecolors='b', label='Original Data Points')
    plt.scatter(X_rec[:, 0], X_rec[:, 1], s=30, facecolors='none', edgecolors='r', label='PCA Reduced Data Points')
    plt.xlabel('x1 [Feature Normalized]')
    plt.ylabel('x2 [Feature Normalized]')
    for i in range(X_norm.shape[0]):
        plt.plot([X_norm[i, 0], X_rec[i, 0]], [X_norm[i, 1], X_rec[i, 1]], 'k--')
    plt.grid(True)
    plt.legend()
    mat_face = sio.loadmat('ex7faces.mat')
    X_face = mat_face['X']
    pdfd.display_data(X_face, 10, 10)
    X_face_norm, mu_face, sigma_face = fn.feature_normalize(X_face)
    U_face, S_face, V_face = prp.run_pca(X_face_norm)
    pdfd.display_data(U_face[:, :36].T, 6, 6)
    Z_face = ppd.project_data(X_face_norm, U_face, k=36)
    X_rec_face = prd.recover_data(Z_face, U_face, k=36)
    pdfd.display_data(X_rec_face, 10, 10)
