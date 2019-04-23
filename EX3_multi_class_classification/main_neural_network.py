#!/usr/bin/python3
# -*- coding:utf-8 -*-
import scipy.io as sio
import numpy as np
import predict_nn as pn


if __name__ == '__main__':
    data = sio.loadmat('ex3data1.mat')
    X = data['X']
    Y = data['y']
    weights = sio.loadmat('ex3weights.mat')
    theta1 = weights['Theta1']
    theta2 = weights['Theta2']
    print(theta1.shape)
    print(theta2.shape)
    p = pn.predict(theta1, theta2, X)
    p = p.reshape((X.shape[0], 1))
    print('Accuracy : {:f}%'.format((np.mean(p==Y)*100)))
