#!/usr/bin/python3
# -*- coding:utf-8 -*-
from scipy.special import expit
import numpy as np

def sigmoid_gradient(x):
    # sigmoid gradient formula:
    # g(x) = 1/(1+exp(-x))
    # g'(x) = g(x)*(1-g(x))
    sg = expit(x)*(1-expit(x))
    return sg