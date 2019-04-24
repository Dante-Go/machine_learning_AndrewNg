#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import linear_regression_cost_func as lrcf
import train_linear_reg as tlr
import learning_curve as lc
import poly_feature as pf
import features_normalization as fn
import plot_fit as pltf
import validate_curve as vc

if __name__ == '__main__':
    data = sio.loadmat('ex5data1.mat')
    X = data['X']
    Y = data['y']
    X_val = data['Xval']
    Y_val = data['yval']
    X_test = data['Xtest']
    Y_test = data['ytest']
    m, n = X.shape
    fig1 = plt.figure()
    plt.plot(X, Y, 'rx', markersize=10, linewidth=1.5)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam(y)')
    fig1.show()
    #
    X = np.column_stack((np.ones((m, 1)), X))
    initial_theta = np.array(np.ones((n+1, 1)))
    lambda_reg = 0.1
    cost, grad = lrcf.linear_cost_func(initial_theta, X, Y, lambda_reg)
    theta = tlr.train_linear_reg(X, Y, lambda_reg)
    print(theta)
    t_y = np.dot(X, theta)
    t_x = X[:, 1:]
    plt.plot(t_x, t_y, '--', linewidth=1)
    plt.show()
    val_m, val_n = X_val.shape
    X_val = np.column_stack((np.ones((val_m, 1)), X_val))
    error_train, error_val = lc.learning_curve(X, Y, X_val, Y_val, 0.1)
    fig2 = plt.figure()
    p1, p2 = plt.plot(range(m), error_train, range(m), error_val)
    plt.xlabel('Learning curve for linear regression')
    plt.ylabel('Number of training examples')
    plt.legend((p1, p2), ('Train', 'Cross validation'))
    fig2.show()
    #
    p = 8
    x_poly = pf.poly_feature(X[:, 1:], p)
    x_poly, mu, sigma = fn.features_normalize(x_poly)
    m_poly, n_poly = x_poly.shape
    x_poly = np.column_stack((np.ones((m_poly, 1)), x_poly))
    x_poly_test = pf.poly_feature(X_test, p)
    x_poly_test = x_poly_test - mu
    x_poly_test = x_poly_test / sigma
    x_poly_test = np.column_stack((np.ones((x_poly_test.shape[0], 1)), x_poly_test))
    x_poly_val = pf.poly_feature(X_val[:, 1:], p)
    x_poly_val = x_poly_val - mu
    x_poly_val = x_poly_val / sigma
    x_poly_val = np.column_stack((np.ones((x_poly_val.shape[0], 1)), x_poly_val))
    #
    lambda_poly = 0
    theta = tlr.train_linear_reg(x_poly, Y, lambda_poly)
    plt.close()
    plt.figure(1)
    plt.plot(X[:, 1:], Y, 'rx', markersize=10, linewidth=1.5)
    pltf.plot_fit(min(X[:, 1:]), max(X[:, 1:]), theta, mu, sigma, p)
    plt.xlabel('Change in water level(x)')
    plt.ylabel('Water flowing out of dam(y)')
    plt.title('Polynomial Regression Fit ( lambda = 0)')
    plt.figure(2)
    error_train_poly, error_val_poly = lc.learning_curve(x_poly, Y, x_poly_val, Y_val, lambda_poly)
    p3, p4 = plt.plot(range(m), error_train_poly, range(m), error_val_poly)
    plt.title('Polynomial regression learning curve (lambda = 0)')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend((p3, p4), ('Train', 'Cross Validation'))
    plt.show()
    # validate curve
    plt.close('all')
    lambda_vec, error_train_vc, error_val_vc = vc.validate_curve(X, Y, X_val, Y_val)
    plt.title('validate curve')
    plt.xlabel('lambda')
    plt.ylabel('error')
    p5, p6 = plt.plot(lambda_vec, error_train_vc*1000000, lambda_vec, error_val_vc*1000000)
    plt.legend((p5, p6), ('Train', 'cross validation'))
    plt.show()

