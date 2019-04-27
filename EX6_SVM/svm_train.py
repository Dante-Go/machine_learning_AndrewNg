#!/usr/bin/python3
# -*- coding:utf-8 -*-
from sklearn import svm
import gaussian_precompute_maxtrix as gpm


def svm_train(x, y, c, kernelFunction, tol=1e-3, max_passes=-1, sigma=0.1):
    y = y.flatten()
    if kernelFunction == 'linear':
        clf = svm.SVC(C=c, kernel='linear', tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(x, y)
    elif kernelFunction == 'gaussian':
        clf = svm.SVC(C=c, kernel='precomputed', tol=tol, max_iter=max_passes, verbose=2)
        return clf.fit(gpm.gaussian_matrix(x, x, sigma), y)
