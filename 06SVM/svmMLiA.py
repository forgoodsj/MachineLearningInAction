#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 2019/3/12
# __author__ = 'jun'
import random
from numpy import *


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    # i是第一个alpha下标， m是alpha数量
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
        # 函数值不等于输入值i，就会随机选择。
    return j


def clipAlpha(aj, H, L):
    # 用于调整大于H或者小于L的alpha值
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  # 简化版SMO
    # 输入数据集，类别标签，常数C，容错率，和最大循环
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()  #得到列向量
    b = 0
    m, n = shape(dataMatrix)  #得到矩阵的行和列
    alphas = mat(zeros((m, 1)))
    iter = 0  #记录遍历次数
    while (iter < maxIter):
        alphaPairsChanged = 0  #每次循环先预设为0，然后用户记录alpha是否已经优化
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b  # 通过alphas 得到预测的类别
            Ei = fXi - float(labelMat[i])  # 预测结果和真实结果的误差
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or (
                (labelMat[i] * Ei > toler) and (alphas[i] > 0)):  # 选择第一个alpha,如果误差值大则可以对alpha进行优化
                j = selectJrand(i, m)  # 随机选择第二个alpha
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])  # 计算误差
                alphaIold = alphas[i].copy()  #复制一个方便后续比较
                alphaJold = alphas[j].copy()
                if ((labelMat[i] * Ei != labelMat[j])):
                    L = max(0, alphas[j] - alphas[i])  #计算L和H，用户保证alpha在0和C之间
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")  #不做任何变化，开始下个循环
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[
                                                                                                            j,
                                                                                                            :] * dataMatrix[
                                                                                                                 j,
                                                                                                                 :].T  #alpha[j]的最优修改量
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta  # 修改j
                alphas[j] = clipAlpha(alphas[j], H, L) #进行调整
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  #对i进行修改，修改量与J相同，但相反

                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                                                                                                                  j] * (
                                                                                                              alphas[
                                                                                                                  j] - alphaJold) * dataMatrix[
                                                                                                                                    i,
                                                                                                                                    :] * dataMatrix[
                                                                                                                                         j,
                                                                                                                                         :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                                                                                                                  j] * (
                                                                                                              alphas[
                                                                                                                  j] - alphaJold) * dataMatrix[
                                                                                                                                    j,
                                                                                                                                    :] * dataMatrix[
                                                                                                                                         j,
                                                                                                                                         :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d  i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


'''
完整版Platt SMO支持函数
'''


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros(self.m, 1))
        self.b = 0
        self.eCache = mat(zeros(self.m, 2))  # 误差缓存


def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):  # 内循环中的启发式方法
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


dataArr, labelArr = loadDataSet('testSet.txt')
b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print(b,alphas)
