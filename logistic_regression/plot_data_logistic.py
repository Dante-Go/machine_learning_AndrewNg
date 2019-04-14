#!/usr/bin/python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plot_data_logistic(x, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    plt.scatter(np.array(x[pos, 0]), np.array(x[pos, 1]), marker='+', linewidths=2, label='admitted')
    plt.scatter(np.array(x[neg, 0]), np.array(x[neg, 1]), marker='o', s=7, label='not admitted')
    plt.xlabel('exam 1 score')
    plt.ylabel('exam 2 score')
    plt.legend()
    plt.show()
    return plt


def plot_boundary(theta, x):
    if np.size(x, axis=1) <= 3:
        plot_x = np.mat([np.min(x[:, 1]), np.max(x[:, 1])])
        plot_y = (-1/theta[2, 0])*(theta[1, 0]*plot_x + theta[0, 0])
        # print(plot_x.tolist())
        # print(plot_y[0][0])
        plt.plot(np.array(plot_x)[0], np.array(plot_y)[0])
        plt.show()
    return plt

