#!/usr/bin/python3
# -*- coding:utf-8 -*-
import scipy.io as sio
import plot_data as pd
import svm_train as svmt
import visualize_boundary_linear as vbl

if __name__ == '__main__':
    data = sio.loadmat('ex6data1.mat')
    X = data['X']
    Y = data['y']
    plt = pd.plot_data(X, Y)
    C = 1
    model = svmt.svm_train(X, Y, C, 'linear', 1e-3, 20)
    vbl.visualize_boundary_linear(X, Y, model)
    # plt.close()

