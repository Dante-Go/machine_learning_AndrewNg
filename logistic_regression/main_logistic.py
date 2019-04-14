#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

import hypothesis_logistic as hf
import plot_data_logistic as pdl
import cost_func_logistic as cfl

if __name__ == '__main__':
    path = 'ex2data1.txt'
    data = pd.read_csv(path)
    data = np.mat(data)
    m, n = np.shape(data)
    x_n = n-1
    x_m = m
    X = data[:, 0:x_n]
    Y = data[:, x_n]
    plt = pdl.plot_data_logistic(X, Y)
    X = np.hstack((np.ones((m, 1)), X))
    theta = np.zeros((x_n+1, 1))
    # print(cfl.cost_function(theta, X, Y))
    # print(cfl.gradient(theta, X, Y))
    theta = cfl.gradient_descent(X, Y, theta, 0.005, 1500000)
    # theta = np.mat([-25, 0.2, 0.2]).T
    pdl.plot_boundary(theta, X)
    print(theta)
    new_x = np.mat([1, 3, 13])
    admitted = hf.predict(new_x, theta)
    print(admitted)
    print(hf.predict(np.mat([1, 89, 87]), theta))
