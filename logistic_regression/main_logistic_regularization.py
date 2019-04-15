#!/usr/bin/python3
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import plot_data_logistic as pdl
import map_features as mf

if __name__ == '__main__':
    data = np.mat(pd.read_csv("ex2data2.txt"))
    X = data[:, :2]
    Y = data[:, 2]
    plt = pdl.plot_data_logistic(X, Y)
    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')
    X1 = X[:, 0]
    X2 = X[:, 1]
    X = mf.map_features(X1, X2)
    print(X.shape)

