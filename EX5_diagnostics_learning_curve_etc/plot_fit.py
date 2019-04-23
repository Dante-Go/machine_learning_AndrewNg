#!/usr/bin/python3
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import poly_feature as pf


def plot_fit(x_min, x_max, theta, mu, sigma, p):
    x = np.array(np.arange(x_min-15, x_max+25, 0.05))
    x_poly = pf.poly_feature(x, p)
    x_poly = x_poly - mu
    x_poly = x_poly / sigma
    x_poly = np.column_stack((np.ones((x_poly.shape[0], 1)), x_poly))
    plt.plot(x, np.dot(x_poly, theta), 'b--', linewidth=2)
