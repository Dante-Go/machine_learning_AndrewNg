#!/usr/bin/python3
# -*-coding:utf-8-*-

# theta = theta - alpha * cost'(theta)
# theta = theta - alpha * ((1/m)*((hypothesis(theta*x)-y)*x))
import costFunc_linear
import hypothesis_linear as hFunc


def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        theta = theta - (alpha/m)*x.T*(hFunc.hypothesis_func(x, theta) - y)
    return theta

