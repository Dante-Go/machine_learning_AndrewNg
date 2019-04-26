#!/usr/bin/python3
# -*- coding:utf-8 -*-
from sklearn import svm


def svm_train(x, y, c, kernelFunction, tol=1e-3, max_passes=-1, sigma=0.1):
    y = y.flatten()
    if kernelFunction == 'linear':
        clf = svm.SVC(C=c, kernel='linear', tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(x, y)
