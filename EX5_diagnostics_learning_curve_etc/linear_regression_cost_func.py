#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def linear_cost_func(theta, x, y, lambda_reg):
    m, n = x.shape
    theta = np.reshape(theta, (n, 1))
    cost = (1./(2*m)) * np.sum(np.power((np.dot(x, theta) - y), 2))
    grad = (1./m) * np.dot(x.T, (np.dot(x, theta) - y))
    cost_reg = cost + (lambda_reg/(2*m)) * np.sum(np.power(theta[1:], 2))
    grad_reg = grad + (lambda_reg/m)*(np.concatenate((np.zeros((1, 1)), theta[1:])))
    return cost_reg, grad_reg
