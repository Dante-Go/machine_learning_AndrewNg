#!/usr/bin/python3
# -*- coding:utf-8 -*-
import spam_read_file as srf
import spam_process_email as spe
import spam_email_features as sef
import scipy.io as sio
import svm_train as st
import spam_get_vocab_list as sgvl

if __name__ == '__main__':
    file_contents = srf.read_file('emailSample1.txt')
    word_indices = spe.process_email(file_contents)
    features = sef.email_features(word_indices)
    mat = sio.loadmat('spamTrain.mat')
    X = mat['X']
    Y = mat['y']
    C = 0.1
    model = st.svm_train(X, Y, C, 'linear')
    p = model.predict(X)
    mat_test = sio.loadmat('spamTest.mat')
    X_test = mat_test['Xtest']
    Y_test = mat_test['ytest']
    p_test = model.predict(X_test)
    w = model.coef_[0]
    indices = w.argsort()[::-1][:15]
    lst_vocabulary = sorted(sgvl.get_vocabulary_list().keys())
    for idx in indices:
        print('{} ({})'.format(lst_vocabulary[idx], float(w[idx])))
