#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import svm_train as st
import gaussian_precompute_maxtrix as gpm


def dataset3_params(x, y, xval, yval):
    c = 1
    sigma = 0.3
    c_s = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]).T
    sigma_s = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]).T
    err_c_s = np.zeros((len(c_s)*len(sigma_s), 3))
    row_inx = 0
    for i in range(len(c_s)):
        for j in range(len(sigma_s)):
            model = st.svm_train(x, y, c_s[i], 'gaussian', sigma=sigma_s[j])
            predictions = model.predict(gpm.gaussian_matrix(xval, x))
            err_c_s[row_inx, 0] = np.mean((predictions != yval).astype(int))
            err_c_s[row_inx, 1] = c_s[i]
            err_c_s[row_inx, 2] = sigma_s[j]
            row_inx = row_inx + 1
    row = err_c_s.argmax(axis=0)
    ret_c = err_c_s[row[0], 1]
    ret_sigma = err_c_s[row[0], 2]
    return ret_c, ret_sigma
