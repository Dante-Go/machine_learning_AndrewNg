#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np

import hypothesis_logistic as hl


def cost_func_reg(theta, x, y, lambda_reg, return_grad=False):
    m, n = x.shape
    theta  = theta.reshape((n,1))
    _theta = np.array(theta)
    _theta = _theta[1:]
    cost = (-1/m)*(y.T * np.log(hl.sigmoid(x*theta)) + (1-y).T * np.log(1 - hl.sigmoid(x*theta))) + (lambda_reg/(2*m))*(np.sum(np.power(_theta, 2)))
    grad_reg = (1/(2*m)) * (x.T * (hl.sigmoid(x*theta) - y)) + (lambda_reg/m) * theta
    grad = (1/(2*m)) * (x.T * (hl.sigmoid(x*theta) - y))
    grad_reg[0] = grad[0]
    if return_grad==True:
        return cost, grad_reg.flatten()
    elif return_grad==False:
        return cost
