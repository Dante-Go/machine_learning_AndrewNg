#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import scipy.io as sio
import display_data as dd
import lr_cost_function as lcf
import one_vs_all as ova
import predict_one_vs_all as preova


if __name__ == '__main__':
    input_layer_size = 400
    num_labels = 10
    data = sio.loadmat('ex3data1.mat')
    X = data["X"]
    Y = data["y"]
    m, n = X.shape
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]
    # dd.display_data(sel)
    theta_t = np.array([-2, -1, 1, 2]).reshape((4, 1))
    X_t = np.hstack((np.ones((5, 1)), np.reshape(np.array(range(1, 16)), (5, 3))/10))
    Y_t = np.array([1, 0, 1, 0, 1]).reshape((5, 1))
    lambda_t = 3
    J, grad = lcf.cost_function(theta_t, X_t, Y_t, lambda_t, True)
    print(J)
    print(grad)
    lambda_t = 0.1
    all_theta = ova.one_vs_all(X, Y, num_labels, lambda_t)
    print(all_theta)
    pre = preova.preict_one_vs_all(all_theta.T, X)
    pre = pre.reshape((m, 1))
    print(pre)
    print('Accuracy: {:f}%'.format((np.mean(pre == Y%10)*100)))

