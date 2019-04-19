#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
from scipy.special import expit


def predict(theta1, theta2, x):
    m, n = x.shape
    x = np.column_stack((np.ones((m,1)), x))
    z1 = np.dot(x, theta1.T)
    print(z1.shape)
    a1 = expit(z1)
    m2, n2 = a1.shape
    a1 = np.column_stack((np.ones((m2,1)), a1))
    z2 = np.dot(a1, theta2.T)
    print(z2.shape)
    a2 = expit(z2)
    pre = np.argmax(a2, axis=1)
    return pre+1
