#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def map_features(x1, x2):
    degree = 6
    out = np.ones(x1.shape[0])
    x1 = np.array(x1)
    x2 = np.array(x2)
    for i in range(1, degree + 1):
        for j in range(i+1):
            col = np.power(x1, (i-j)) * np.power(x2, j)
            out = np.column_stack((out, col))
    return np.mat(out)

#def map_features(X1, X2):
#    degree = 6
#    out = np.ones(X1.shape[0])
#    X1 = np.array(X1)
#    X2 = np.array(X2)
#    for i in range(degree):
#        for j in range(i):
#            new_column = (X1 ** (i - j)) * (X2 ** j)
#            out = np.column_stack((out, new_column))
#    return np.mat(out)