#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def email_features(word_indices):
    n = 1899
    x = np.zeros((n, 1))
    for i in word_indices:
        x[i] = 1
    return x
