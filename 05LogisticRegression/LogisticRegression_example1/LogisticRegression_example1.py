#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 2019/3/9
# __author__ = 'jun'

from numpy import *


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 1.0为常数的系数
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):  # 梯度上升
    # dataMatIn是100*3的矩阵，代表不同的特征的值 包含X1 X2和X0（常数特征）
    dataMatrix = mat(dataMatIn)  # 将数据转换为numpy矩阵
    labelMat = mat(classLabels).transpose()  # 转置
    m, n = shape(dataMatrix)
    alpha = 0.001  # 步频
    maxCyclee = 500
    weights = ones((n, 1))
    for k in range(maxCyclee):
        h = sigmoid(dataMatrix * weights)  # 矩阵相乘，h是一个列向量.(100*3) *(1*3)=(100*1)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def stocGradAscent(dataMatrix, classLabels, numIter=150):  # 随机梯度上升法
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # alpha步频每次迭代前进行调整
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本来更新回归系数
            h = sigmoid(sum(dataMatrix[randIndex] * weights))  # h是一个数字
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])  # 删除被选中样本
    return weights


# dataArr, labelMat = loadDataSet()
# a = gradAscent(dataArr, labelMat)
# print(a)

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))  # 带入sigmoid函数
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)  # 训练集的特征
        trainingLabels.append(float(currLine[21]))  # 训练集的标签
    trainWeights = stocGradAscent(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


multiTest()
