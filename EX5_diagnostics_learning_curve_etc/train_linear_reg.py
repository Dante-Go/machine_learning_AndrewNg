#!/usr/bin/python3
# -*- coding:utf-8 -*-
import scipy.optimize as sopt
import numpy as np
import linear_regression_cost_func as lrcf

def train_linear_reg(x, y, lambda_reg):
    m, n = x.shape
    initial_theta = np.zeros((n, 1))
    options = {'maxiter':200, 'disp':True}
    myargs = (x, y, lambda_reg)
    ret = sopt.minimize(lrcf.linear_cost_func, x0=initial_theta, args=myargs, options=options, method='L-BFGS-B', jac=True)
    theta = ret['x']
    return  theta
