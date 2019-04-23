#!/usr/bin/python3
# -*-coding:utf-8-*-

import numpy as np


def feature_normalize(x):
    mean_value = np.mean(x, 0)
    sigma = np.std(x, 0)
    x_norm = (x - mean_value) / sigma
    return x_norm, mean_value, sigma

