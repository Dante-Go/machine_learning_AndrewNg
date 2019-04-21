#!/usr/bin/python3
# -*- coding:utf-8 -*-
import scipy.io as sio
import scipy.optimize as sop
import numpy as np

import predict_nn as pnn
import cost_func_neural_network as cfnn
import sigmoid_gradient as sg
import random_initialize_weights as riw

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
    lambda_reg = 1
    cost, grad = cfnn.cost_func(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_reg)
    print(cost)
    init_theta1 = riw.random_initialize_weights(input_layer_size, hidden_layer_size)
    init_theta2 = riw.random_initialize_weights(hidden_layer_size, num_labels)
    initial_params = np.concatenate((init_theta1.reshape(init_theta1.size, order='F'),
                                     init_theta2.reshape(init_theta2.size, order='F')))
    max_iter = 50
    lambda_reg = 0.1
    my_args = (input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_reg)
    ret = sop.minimize(cfnn.cost_func, x0=initial_params, args=my_args, options={'disp':True, 'maxiter':max_iter},
                       method='L-BFGS-B', jac=True)
    nn_ret_params = ret['x']
    ret_theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1),
                            order='F')
    ret_theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1),
                            order='F')
    pred = pnn.predict(ret_theta1, ret_theta2, X)
    print('Training set Accuracy: {:f}%'.format((np.mean(pred == Y)*100)))
