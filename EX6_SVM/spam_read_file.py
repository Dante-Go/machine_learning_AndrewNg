#!/usr/bin/python3
# -*- coding:utf-8 -*-


def read_file(path):
    try:
        with open(path, 'r') as fid:
            contents = fid.read()
    except:
        contents = ''
    return contents
