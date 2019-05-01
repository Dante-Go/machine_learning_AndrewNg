#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def run_pca(x):
    m, n = x.shape
    Sigma = (1/m)*(x.T@x)
    U, S, V= np.linalg.svd(Sigma)
    return U, S, V
