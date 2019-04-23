#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
from scipy.special import expit


def preict_one_vs_all(all_theta, x):
    m, n = x.shape
    x = np.column_stack((np.ones((m, 1)), x))
    pre = np.argmax(expit(np.dot(x, all_theta)), axis=1)
    return pre