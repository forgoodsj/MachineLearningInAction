#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 2019/2/26
# __author__ = 'jun'
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group,labels = createDataSet()
# print(group,labels)

def classify0(inX, dataSet, labels, k):
    #输入向量inX, 训练样本集dataSet, 标签集 labels， k选取前k个
    dataSetSize = dataSet.shape[0]
    #距离计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  #tile重复某个数组n次，变成一个n*dataSetSize的数组
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)#按横轴加和
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() #将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    #选择最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #标签+1
    #排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)#返回前k个中各个标签占了几个位置，从大到小排
    print(sortedClassCount)
    return sortedClassCount[0][0]

result = classify0([0,0], group, labels, 3)
print(result)
