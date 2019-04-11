#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

import featureNormalize as fn
import gradientDescent as gd


if __name__ == '__main__':
    data = pd.read_csv('ex1data2.txt')
    data = np.mat(data)
    X = data[:, 0:2]
    Y = data[:, 2]
    m = len(Y)
    n = np.size(X, 1)
    [X, mean_value, sigma] = fn.feature_normalize(X)
    X = np.hstack((np.ones((m, 1)), X))
    alpha = 0.01
    theta = np.mat(np.zeros((n+1, 1)))
    iterations = 300
    theta = gd.gradient_descent(X, Y, theta, alpha, iterations)
    new_x = np.mat([1650, 3])
    new_x = (new_x - mean_value) / sigma
    new_x = np.hstack((np.ones((1, 1)), new_x))
    predict = new_x * theta
    print(predict)


