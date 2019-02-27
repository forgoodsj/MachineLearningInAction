#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 2019/2/27
# __author__ = 'jun'

from numpy import *
import operator
from os import listdir


def img2vector(filename):  # 将图像转换为向量
    # 该函数创建1*1024的Numpy数组，然后打开给定文件，循环读出出文件的前32行，并将每行的32个字符存在数组中
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


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
    # print(sortedClassCount)
    return sortedClassCount[0][0]


def handwritingClassTest():
    hwLabels = []
    # 获取目录内容
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 文件名格式 0_0.txt
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)  # 将文本转化为向量存入训练集中
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


handwritingClassTest()
