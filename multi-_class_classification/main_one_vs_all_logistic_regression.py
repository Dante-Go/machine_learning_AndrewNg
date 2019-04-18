#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import scipy.io as sio
import display_data as dd


if __name__ == '__main__':
    input_layer_size = 400
    num_labels = 10
    data = sio.loadmat('ex3data1.mat')
    X = data["X"]
    Y = data["y"]
    m, n = X.shape
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]
    dd.display_data(sel)