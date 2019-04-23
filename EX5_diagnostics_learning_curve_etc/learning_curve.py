#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import train_linear_reg as tlr
import linear_regression_cost_func as lrcf


def learning_curve(x, y, xval, yval, lambda_reg):
    m, n = x.shape
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    for i in range(1, m+1):
        theta = tlr.train_linear_reg(x[:i], y[:i], lambda_reg)
        error_train[i-1], _ = lrcf.linear_cost_func(theta, x[:i], y[:i], 0)
        error_val[i-1], _ = lrcf.linear_cost_func(theta, xval, yval, 0)
    return error_train, error_val
