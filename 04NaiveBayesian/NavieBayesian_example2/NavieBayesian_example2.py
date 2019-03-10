#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 2019/3/6
# __author__ = 'jun'

import feedparser
import jieba
import re

pattern = re.compile(u'[\u4E00-\u9FA5]|[0-9A-Za-z]')
# a = ''.join(pattern.findall('中国u你好'))
# print(a)
ny = feedparser.parse('http://feed.cnblogs.com/blog/sitehome/rss')
for i in ny['entries']:
    seg_list = jieba.lcut_for_search(''.join(pattern.findall(i['title'])))
    print(i['title'])
    print(seg_list)


def calcMostFreq(vocabList, fullText):
    pass
