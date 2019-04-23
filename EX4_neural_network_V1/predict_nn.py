#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
from scipy.special import expit


def predict(theta1, theta2, x):
    m, n = x.shape
    a0 = np.column_stack((np.ones((m, 1)), x))
    print('x shape: %d %d' % x.shape)
    print('theta1 shape: %d %d' % theta1.shape)
    z1 = np.dot(a0, theta1.T)
    a1 = expit(z1)
    a1 = np.column_stack((np.ones((a1.shape[0], 1)), a1))
    print('a1 shape: %d %d' % a1.shape)
    print('theta2 shape: %d %d' % theta2.shape)
    z2 = np.dot(a1, theta2.T)
    a2 = expit(z2)
    y = np.argmax(a2, axis=1)
    return (y+1).reshape((x.shape[0], 1))
