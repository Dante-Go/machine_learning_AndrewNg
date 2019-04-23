#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
from scipy.special import expit

import sigmoid_gradient as sg


def cost_func(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg):
    theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1),
                        order='F')
    theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1),
                        order='F')
    # print('theta1 shape: (%d, %d)' % theta1.shape)
    # print('theta2 shape: (%d, %d)' % theta2.shape)
    m, n = x.shape
    a0 = np.column_stack((np.ones((m, 1)), x))
    z1 = np.dot(a0, theta1.T)
    a1 = expit(z1)
    a1 = np.column_stack((np.ones((a1.shape[0], 1)), a1))
    z2 = np.dot(a1, theta2.T)
    a2 = expit(z2)
    cost = 0
    # octave : 1-1, ... 9-9, 10 - 0
    # python : 0-1, ... 8-9, 9 - 0
    tc = y
    y_c = np.zeros((m, num_labels))
    for i in range(m):
        y_c[i, tc[i] - 1] = 1
    # for i in range(m):
    #    cost += np.sum(y_c[i] * np.log(a2[i]) + (1 - y_c[i]) * np.log(1 - a2[i]))
    cost = np.sum(np.sum((y_c * np.log(a2)) + ((1 - y_c)*np.log(1 - a2))))
    cost = (-1/m) * cost
    reg = (lambda_reg / (2 * m)) * (
                np.sum(np.sum(np.power(theta1[:, 1:], 2))) + np.sum(np.sum(np.power(theta2[:, 1:], 2))))
    cost = cost + reg

    Delta1 = 0
    Delta2 = 0
    T = np.column_stack((np.ones((m, 1)), x))
    for t in range(m):
        # step 1
        a_1 = T[t]
        z_2 = np.dot(a_1, theta1.T)
        a_2 = expit(z_2)
        # a_2 = np.column_stack((np.ones((a_2.shape[0], 1)), a_2))
        a_2 = np.concatenate((np.array([1]), a_2))
        z_3 = np.dot(a_2, theta2.T)
        a_3 = expit(z_3)
        # step 2
        delta_3 = np.zeros((num_labels))
        for j in range(num_labels):
            y_t = y_c[t]
        delta_3 = a_3 - y_t
        # step 3
        # delta_2 = (theta2[:, 1:].T * delta_3) * sg.sigmoid_gradient(z_2)
        delta_2 = (np.dot(theta2[:, 1:].T, delta_3))
        delta_2 = delta_2 * sg.sigmoid_gradient(z_2)
        # step 4
        # Delta1 = Delta1 + delta_2 * a_1.T
        # Delta2 = Delta2 + delta_3 * a_2.T
        Delta1 = Delta1 + np.outer(delta_2, a_1)
        Delta2 = Delta2 + np.outer(delta_3, a_2)
    reg_1 = (lambda_reg/m)*np.column_stack((np.zeros((theta1.shape[0], 1)), theta1[:, 1:]))
    Theta_grad1 = (1/m)*Delta1 + reg_1
    # reg_2 = (lambda_reg/m) * np.hstack((np.zeros((theta2.shape[0], 1)), theta2[:, 1:]))
    Theta_grad2 = (1/m)*Delta2 + (lambda_reg/m) * np.hstack((np.zeros((theta2.shape[0], 1)), theta2[:, 1:]))

    grad = np.concatenate((Theta_grad1.reshape(Theta_grad1.size, order='F'), Theta_grad2.reshape(Theta_grad2.size, order='F')))

    return cost, grad
