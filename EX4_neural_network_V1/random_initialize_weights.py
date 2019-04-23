#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np


def random_initialize_weights(l_in, l_out):
    W = np.zeros((l_out, l_in+1))
    # epsilon_init = sqrt(6) / sqrt(l_in, l_out)
    epsilon_init = 0.12
    W = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
    return W
