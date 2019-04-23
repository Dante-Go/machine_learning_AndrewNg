#!/usr/bin/python3
# -*-coding:utf-8-*-

import matplotlib.pyplot as plt


def plot_scatter(x, y):
    plt.scatter(x, y, s=10, c='r', marker='x')
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()
    return plt
