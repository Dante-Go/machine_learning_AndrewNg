#!/usr/bin/python3
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import plot_data as pd
import numpy as np


def visualize_boundary_linear(x, y, model):
    w = model.coef_[0]
    b = model.intercept_[0]
    xp = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
    yp = -(w[0] * xp + b) / w[1]
    plt.plot(xp, yp, 'b-')
    pd.plot_data(x, y)
