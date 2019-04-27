#!/usr/bin/python3
# -*- coding:utf-8 -*-


def get_vocabulary_list():
    with open('vocab.txt', 'r') as fid:
        lst_vacabulary = {}
        for line in fid.readlines():
            i, word = line.split()
            lst_vacabulary[word] = int(i)
    return lst_vacabulary
