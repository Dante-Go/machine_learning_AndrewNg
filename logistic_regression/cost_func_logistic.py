#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import hypothesis_logistic as hfl


def cost_function(theta, x, y):
    m = np.size(x, axis=0)
    p = y.T * np.log(hfl.sigmoid(x*theta)) + (1-y).T * np.log((1 - hfl.sigmoid(x*theta)))
    cost = (-1/m) * p
    return cost


def gradient(theta, x, y):
    m = np.size(x, axis=0)
    grad = (x.T * (hfl.sigmoid(x*theta) - y)) / m
    return grad


def gradient_descent(x, y, theta, alpha, iterations):
    for i in range(iterations):
        theta = theta - alpha * gradient(theta, x, y)
        # if (i % 1000) == 0:
        #   print(cost_function(x, y, theta))
    return theta
