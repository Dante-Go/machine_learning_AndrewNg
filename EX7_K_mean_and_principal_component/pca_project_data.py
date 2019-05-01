#!/usr/bin/python3
# -*- coding:utf-8 -*-


def project_data(x, u, k):
    Z = x @ u[:, :k]
    return Z
