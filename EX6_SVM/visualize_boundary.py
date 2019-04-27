#!/usr/bin/python3
# -*- coding:utf-8 -*-
import plot_data as pd
import numpy as np
import gaussian_precompute_maxtrix as gpm


def visualize_boundary(x, y, model, varargin=0):
    plt = pd.plot_data(x, y)
    x1plot = np.linspace(min(x[:, 0]), max(x[:, 0]), 100).T
    x2plot = np.linspace(min(x[:, 1]), max(x[:, 1]), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_x = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = model.predict(gpm.gaussian_matrix(this_x, x, 0.1))
    plt.contour(X1, X2, vals, color='blue', levels=[0, 0])
    plt.show()

