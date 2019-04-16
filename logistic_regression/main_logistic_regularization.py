#!/usr/bin/python3
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

import plot_data_logistic as pdl
import map_features as mf
import cost_func_logistic_regularization as cflr
import hypothesis_logistic as hl

import scipy.optimize as op



if __name__ == '__main__':
    data = np.mat(pd.read_csv("ex2data2.txt"))
    X = data[:, :2]
    Y = data[:, 2]
    plt = pdl.plot_data_logistic(X, Y)
    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')
    X1 = X[:, 0]
    X2 = X[:, 1]
    X = mf.map_features(X1, X2)
    m, n = X.shape
    initial_theta = np.zeros((n, 1))
    lambda_reg = 1
    theta = op.fmin_bfgs(cflr.cost_func_reg, x0=initial_theta, args=(X, Y, lambda_reg))
    print(theta)
    pdl.plot_boundary(theta, X)
    p = hl.predict(X, theta)
    print("Train Accuracy: {:f}".format(np.mean(p == Y)*100))

