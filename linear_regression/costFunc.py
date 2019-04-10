#!/usr/bin/python3
# -*-coding:utf-8-*-

# cost = (1/2)*(hypothesis(x, theta) - y)^2
import hypothesisFun as hFunc
import numpy as np


def cost_function(x, y, theta):
    cost = (hFunc.hypothesis_func(x, theta) - y).T * (hFunc.hypothesis_func(x, theta) - y)*(1/(2*len(y)))
    return cost
