#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def features_normalize(x):
    m, n = x.shape
    mu = np.mean(x, axis=0)
    x_norm = x - mu
    sigma = np.std(x_norm, axis=0)
    x_norm = x_norm / sigma
    return x_norm, mu, sigma
