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
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # 误差缓存，第一列给出是否有效的标志位，第二列给出的是实际的E值


def calcEk(oS, k):  # 计算E值并返回
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):  # 内循环中的启发式方法，选择内循环的alpha值，目标是选择合适的第二个alpha值以保证在每次优化中采用最大步长
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 首先将Ei第一个标识位设置为有效的。
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  #返回了非零E值对应的alpha值
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):  #选择具有最大步长的j
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):  #计算误差值并存入缓存中。
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):  # 内循环
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
                (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):  # 选择第一个alpha,如果误差值大则可以对alpha进行优化
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()  # 复制一个方便后续比较
        alphaJold = oS.alphas[j].copy()
        if ((oS.labelMat[i] * Ei != oS.labelMat[j])):
            L = max(0, oS.alphas[j] - oS.alphas[i])  # 计算L和H，用户保证alpha在0和C之间
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")  # 不做任何变化，开始下个循环
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T  # alpha[j]的最优修改量
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta  # 修改j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)  # 进行调整
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # 对i进行修改，修改量与J相同，但相反
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
            oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):  # 完整版smo
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d,pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas








dataArr, labelArr = loadDataSet('testSet.txt')
b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
print(b,alphas)
