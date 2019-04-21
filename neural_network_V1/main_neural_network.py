#!/usr/bin/python3
# -*- coding:utf-8 -*-
import scipy.io as sio
import numpy as np

import predict_nn as pnn
import cost_func_neural_network as cfnn
import sigmoid_gradient as sg

if __name__ == '__main__':
    data = sio.loadmat('ex4data1.mat')
    X = data['X']
    Y = data['y']
    # print('X shape: ({:d} {:d})'.format(X.shape[0], X.shape[1]))
    # print('Y shape: ({:d} {:d})'.format(Y.shape[0], Y.shape[1]))
    weights = sio.loadmat('ex4weights.mat')
    theta1 = weights['Theta1']
    theta2 = weights['Theta2']
    # print('theta1 shape: {} {}'.format(theta1.shape[0], theta1.shape[1]))
    # print('theta2 shape: {} {}'.format(theta2.shape[0], theta2.shape[1]))
    # p = pnn.predict(theta1, theta2, X)
    # print(np.mean(p==Y))
    nn_params = np.concatenate((theta1.reshape(theta1.size, order='F'), theta2.reshape(theta2.size, order='F')))
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    lambda_reg = 0
    cost, grad = cfnn.cost_func(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_reg)
    print(cost)
