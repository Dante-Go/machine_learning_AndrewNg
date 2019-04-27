#!/usr/bin/python3
# -*- coding:utf-8 -*-
import spam_get_vocab_list as sgvl
import re
from nltk import PorterStemmer


def process_email(contents):
    lst_vacabulary = sgvl.get_vocabulary_list()
    word_indices = []
    contents = contents.lower()
    contents = re.sub('<[^<>]+>', ' ', contents)
    contents = re.sub('[0-9]+', 'number', contents)
    contents = re.sub('(http|https)://[^\s]*', 'httpaddr', contents)
    contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', contents)
    contents = re.sub('[$]+', 'dollar', contents)
    contents = re.split(r'[@$/#.-:&\*\+=\[\]?!(){},\'\'\">_<;%\s\n\r\t]+', contents)
    l = 0
    for token in contents:
        token = re.sub('[^a-zA-Z0-9]', '', token)
        token = PorterStemmer().stem(token.strip())
        if len(token) < 1:
            continue
        idx = lst_vacabulary[token] if token in lst_vacabulary else 0
        if idx > 0:
            word_indices.append(idx)
        if 1 + len(token) + 1 > 78:
            l = 0
        print('{:s}'.format(token))
        l = l + len(token) + 1
    return word_indices
