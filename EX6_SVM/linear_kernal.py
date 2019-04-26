#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def linear_kernal(x1, x2):
    sim = np.dot(x1, x2.T)
    return sim
