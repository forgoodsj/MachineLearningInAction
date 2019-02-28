#!/user/bin/env python
#coding=utf-8

from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {} #为所有的分类创建字典，统计标签出现次数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries #该分类出现的概率
        shannonEnt -= prob * log(prob,2) #以2为底求对数
    return shannonEnt
    