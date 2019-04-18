#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
from scipy.special import expit


def cost_function(theta, x, y, lambda_r, ret_grad=False):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    reg = (lambda_r/(2*m))*(np.power(theta[1:], 2).sum())
    cost = (-1/m)*(np.dot(y.T, np.log(expit(np.dot(x, theta)))) + np.dot((1-y).T, np.log(1-expit(np.dot(x, theta))))) + reg
    grad = (1/m)*np.dot(x.T, (expit(np.dot(x, theta)) - y)) + (lambda_r/m) * theta
    grad[0] = (1/m)*np.dot(x.T, (expit(np.dot(x, theta)) - y))[0]
    if ret_grad == True:
        return cost, grad.flatten()
    else:
        return cost

