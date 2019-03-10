#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 2019/2/26
# __author__ = 'jun'
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


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


# print(datingDataMat)
# print(datingLabels[0:20])

# 作图
# fig = plt.figure()
# ax = fig.add_subplot(111)#画是1*1中的第1个
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))#第三个参数大小，第四个参数颜色
# plt.show()

def autoNorm(dataSet):  # 归一化
    minVals = dataSet.min(0)  # 0代表列，1代表行，寻找每列最小值，值为1*3的矩阵
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))  # tile函数重复某个数组，dataset为1000*3，所以需要复制minVals
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# normMat, ranges, minVals = autoNorm(datingDataMat)
# print(normMat)

def datingClassTest():  # 测试分类器正确率
    hoRatio = 0.10  # 测试数据集占比
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 读数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 归一化
    m = normMat.shape[0]  # 获取列数
    numTestVecs = int(m * hoRatio)  # 获取测试数量
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m],
                                     20)  # 第i行数据根据剩下90%数据跑出来的结果
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['不喜欢', '有一点喜欢', '超喜欢']
    percentTats = float(input('花在打游戏的时间占比：'))
    iceCream = float(input('每年消耗的冰激凌公升数：'))
    ffMiles = float(input('每年飞行的里程数：'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("看起来你对这个人：", resultList[classifierResult - 1])


classifyPerson()
