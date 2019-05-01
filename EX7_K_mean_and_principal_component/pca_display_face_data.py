#!/usr/bin/python3
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt


def display_data(x, row, col):
    fig, axs = plt.subplots(row, col, figsize=(6, 6))
    for r in range(row):
        for c in range(col):
            axs[r][c].imshow(x[r*col+c].reshape(32, 32).T, cmap='Greys_r')
            axs[r][c].set_xticks([])
            axs[r][c].set_yticks([])
