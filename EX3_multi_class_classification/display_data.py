#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math


def display_data(x, example_width=None):
    plt.close()
    plt.figure()
    if x.ndim == 1:
        x = np.reshape(x, (-1, x.shape[0]))
    if not example_width or not example_width in locals():
        example_width = int(round(math.sqrt(x.shape[1])))
    plt.set_cmap("gray")
    m, n = x.shape
    example_height = n // example_width
    display_rows = int(math.floor(math.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))
    pad = 1
    display_array = -np.ones((pad + int(display_rows * (example_height + pad)), pad + int(display_cols * (example_width + pad))))
    curr_ex = 1
    for j in range(1, display_rows + 1):
        for i in range(1, display_cols + 1):
            if curr_ex > m:
                break
            max_val = max(abs(x[curr_ex-1, :]))
            rows = pad + int((j - 1) * (example_height + pad)) + np.array(range(example_height))
            cols = pad + int((i - 1) * (example_width + pad)) + np.array(range(example_width))
            display_array[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1] = np.reshape(x[curr_ex-1, :], (example_height, example_width), order="F")//max_val
            curr_ex += 1
        if curr_ex > m:
            break
    h = plt.imshow(display_array, vmin=-1, vmax=1)
    plt.axis('off')
    plt.show(block=False)
    return h, display_array
