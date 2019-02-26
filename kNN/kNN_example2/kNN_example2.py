#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 2019/2/26
# __author__ = 'jun'
from numpy import *
import operator


def classify0(inX, dataSet, labels, k):  # 算法本体
    # 输入向量inX, 训练样本集dataSet, 标签集 labels， k选取前k个
    dataSetSize = dataSet.shape[0]
    # 距离计算
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile重复某个数组n次，变成一个n*dataSetSize的数组
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 按横轴加和
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()  # 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # 选择最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 标签+1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 返回前k个中各个标签占了几个位置，从大到小排
    print(sortedClassCount)
    return sortedClassCount[0][0]


def file2matrix(filename):  # 将文本转换为Numpy的矩阵和向量
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)  # 得到文件行数
    returnMat = zeros((numberOfLines, 3))  # 创建一个0矩阵
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')  # 去掉所有回车符号
        returnMat[index, :] = listFromLine[0:3]  # 取前3个数据（特征数据），存到特征矩阵
        classLabelVector.append(int(listFromLine[-1]))  # 最后一个数据（标签数据），存到标签矩阵
        index += 1
    return returnMat, classLabelVector


datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(datingLabels[0:20])
