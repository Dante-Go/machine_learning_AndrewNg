#!/usr/bin/python3
# -*-coding:utf-8-*-

import pandas as pd
import numpy as np

import plotData_linear as pltd
import gradientDescent_linear as gradientDesc
import hypothesis_linear as hFunc


if __name__ == '__main__':
    path = 'ex1data1.txt'
    data = pd.read_csv(path)
    data = np.mat(data)
    X = data[:, 0]
    Y = data[:, 1]
    plt = pltd.plot_scatter(np.array(X), np.array(Y))
    m = len(Y)
    X = np.hstack((np.ones((m, 1)), X))
    n = np.size(X, axis=1)
    theta = np.zeros((n, 1))
    theta = gradientDesc.gradient_descent(X, Y, theta, 0.005, 15000)
    plt.plot(X[:, 1], hFunc.hypothesis_func(X,theta), 'b--')
    predict_data = np.mat([1, 7])
    predict = hFunc.hypothesis_func(predict_data, theta)*10000
    print(predict)

