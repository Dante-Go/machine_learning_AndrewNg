#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def poly_feature(x, p):
    x_p = x
    for i in range(1, p):
        x_p = np.column_stack((x_p, np.power(x, i+1)))
    return x_p
