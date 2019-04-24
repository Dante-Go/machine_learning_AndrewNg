#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import train_linear_reg as tlr
import linear_regression_cost_func as lrcf


def validate_curve(x, y, x_val, y_val):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    m = lambda_vec.shape[0]
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    for i in range(m):
        lambda_cur = lambda_vec[i]
        theta = tlr.train_linear_reg(x, y, lambda_cur)
        error_train[i], drop1 = lrcf.linear_cost_func(theta, x, y, 0)
        error_val[i], drop2 = lrcf.linear_cost_func(theta, x_val, y_val, 0)
    return lambda_vec, error_train, error_val
