#!/usr/bin/python3
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt


def plot_data(x, y):
    y = y.flatten()
    pos = y == 1
    neg = y == 0
    plt.plot(x[:, 0][pos], x[:, 1][pos], 'k+', markersize=5)
    plt.plot(x[:, 0][neg], x[:, 1][neg], 'yo', markersize=5)
    plt.show()
    return plt
