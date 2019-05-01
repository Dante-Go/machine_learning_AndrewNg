#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def feature_normalize(x):
    m, n = x.shape
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma
