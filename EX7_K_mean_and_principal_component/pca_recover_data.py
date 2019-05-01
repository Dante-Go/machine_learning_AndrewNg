#!/usr/bin/python3
# -*- coding:utf-8 -*-


def recover_data(z, u, k):
    X_rec = z @ u[:, :k].T
    return X_rec
