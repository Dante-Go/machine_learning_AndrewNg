#!/usr/bin/python3
# -*- coding:utf-8 -*-
import scipy.io as sio
import numpy as np
import plot_data as pd
import svm_train as svmt
import visualize_boundary_linear as vbl
import gaussian_kernel as gk
import visualize_boundary as vb
import dataset3_params as d3p

if __name__ == '__main__':
    data = sio.loadmat('ex6data1.mat')
    X = data['X']
    Y = data['y']
    plt = pd.plot_data(X, Y)
    C = 1
    model = svmt.svm_train(X, Y, C, 'linear', 1e-3, 20)
    vbl.visualize_boundary_linear(X, Y, model)
    plt.close()
    # gaussian kernel
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = gk.gaussian_kernel_func(x1, x2, sigma)
    print('Gaussian kernel test (0.324652) : ')
    print(sim)
    data2 = sio.loadmat('ex6data2.mat')
    X2 = data2['X']
    Y2 = data2['y']
    plt = pd.plot_data(X2, Y2)
    C2 = 1
    sigma2 = 0.1
    # model2 = svmt.svm_train(X2, Y2, C2, 'gaussian', sigma=sigma2)
    plt.close()
    #vb.visualize_boundary(X2, Y2, model2)
    # data3
    plt.close()
    data3 = sio.loadmat('ex6data3.mat')
    X3 = data3['X']
    Y3 = data3['y']
    Xval = data3['Xval']
    Yval = data3['yval']
    c3, sigma3 = d3p.dataset3_params(X3, Y3, Xval, Yval)
    model3 = svmt.svm_train(X3, Y3, c3, 'gaussian', sigma=sigma3)
    vb.visualize_boundary(X3, Y3, model3)
