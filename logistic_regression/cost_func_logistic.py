#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import hypothesis_logistic as hfl


def cost_function(theta, x, y):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    p = y.T * np.log(hfl.sigmoid(x*theta)) + (1-y).T * np.log((1 - hfl.sigmoid(x*theta)))
    cost = (-1/m) * p
    return cost


def gradient(theta, x, y):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    grad = (x.T * (hfl.sigmoid(x*theta) - y)) / m
    return grad.flatten()


def gradient_descent(x, y, theta, alpha, iterations):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    for i in range(iterations):
        grad = gradient(theta, x, y).reshape((n, 1))
        theta = theta - alpha * grad
        # if (i % 1000) == 0:
        #   print(cost_function(theta, x, y))
    return np.array(sum(theta.tolist(), []))
