#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import scipy.optimize as op
import lr_cost_function as lcf


def one_vs_all(x, y, num_label, lambda_reg):
    m, n = x.shape
    initial_theta = np.zeros((n+1, 1))
    all_theta = np.zeros((num_label, n+1))
    x = np.column_stack((np.ones((m, 1)), x))
    for i in range(num_label):
        myargs = (x, (y%10==i), lambda_reg, True)
        theta = op.minimize(lcf.cost_function, x0=initial_theta, args=myargs, options={'disp': True, 'maxiter':13}, method='Newton-CG', jac=True)
        all_theta[i, :] = theta["x"]
    return all_theta
