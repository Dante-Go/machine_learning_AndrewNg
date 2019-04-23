#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def sigmoid(z):
    p = 1 / (1 + np.exp(-z))
    return p


def predict(x, theta):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    z = x * theta
    a = sigmoid(z) > 0.5
    return a