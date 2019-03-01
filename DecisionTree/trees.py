#!/user/bin/env python
#coding=utf-8

from math import log
import operator


def calcShannonEnt(dataSet):  # 熵越高越混乱
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


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no'],
               #               [0,-1,'maybe'],
               ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


myDat, labels = createDataSet()


# a = calcShannonEnt(myDat)
# print(a)

def splitDataSet(dataSet, axis, value):  # 按照给定特征划分数据集（把第axis项符合value的向量拿出来，去掉axis项后放入一个新列表）
    # 参数1：待划分数据集，参数2：划分数据集的特征（第n项），参数3：需要返回的标签值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])  # 在列表结尾加入另一个列表的元素
            retDataSet.append(reducedFeatVec)
    return retDataSet


# print(splitDataSet(myDat,0,1))

def chooseBestFeatureToSplit(dataSet):  # 返回哪个特征是最好的用于划分数据集的特征
    numFeatures = len(dataSet[0]) - 1  # 去掉最后一列（因为是标签)
    baseEntropy = calcShannonEnt(dataSet)  # 原来的熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 第i个特征可能有n个标签
        uniqueVals = set(featList)  # 把列表变成集合，创建一个无序不重复元素集
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 按照第i个特征的值是value标签划分数据集（去掉i特征为value的项目）
            prob = len(subDataSet) / float(len(dataSet))  # 这次划分的概率
            newEntropy += prob * calcShannonEnt(
                subDataSet)  # 这里等号右边的熵为去掉value标签后集合的熵(这里面乘以的是这样划分后不同标签的概率），所以还要额外乘这个value出现的概率
        infoGain = baseEntropy - newEntropy  # 以第i个特征划分的信息增益，信息增益是熵的减少，所以划分后的熵越小越好
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回哪个特征是最好的用于划分数据集的特征


# print(chooseBestFeatureToSplit(myDat))
# print(myDat)

def majorityCnt(classList):
    # 使用分类名称的列表，存储了每个标签出现的频率，返回出现次数最多的分类名称
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):  # 创建树的函数代码，递归函数
    # labels 为特征的解释集，例如第一个特征代表什么意思，第二个，第三个
    classList = [example[-1] for example in dataSet]  # 获取数据集中所有的标签
    if classList.count(classList[0]) == len(classList):  # 第一个停止条件：如果数据集中所有的标签相同（第一个标签的数量等于列表长度），则停止划分，并返回
        return classList[0]  # 返回该类标签
    if len(dataSet[0]) == 1:  # 第二个终止条件：如果使用完了所有特征，仍不能将数据集划分成仅包含唯一类别的分组
        return majorityCnt(classList)  # 挑选出现次数最多的标签作为返回值。
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 得到最好的分割的特征
    bestFeatLabel = labels[bestFeat]  # 获得该特征的解释
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # 第bestFeat个特征可能有n个标签
    uniqueVals = set(featValues)
    for value in uniqueVals:  # 遍历该特征所有的标签（surfacing:0,1）
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
                                                  subLabels)  # 拿第一个value去递归，可能就直接退出了或者重复递归
        # print(myTree)
    return myTree  # {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}


'''
需要解释一下这个
第一次打开函数，没有终止，继续发现最佳特征是0（即以 no surfacing）划分。
然后创建一个字典然{no surfacing:{}}
后进入for循环
no surfacing 有两个值0和1，
首先使用no surfacing=0进入for中的递归，发觉终止，返回no，于是{no surfacing:{0:'no',}}
再使用no surfacing=1进入for中递归，没有终止，发觉这次最佳特诊是这次的0（flippers)划分,创建字典{'flippers': {}}
    进入以flippers划分后的for循环，有两个值0和1，
        首先使用flippers=0进入for循环，发觉终止，返回no，于是{'flippers':{0:'no',}}
        再次使用flippers=1进入for循环，发觉终止，返回yes，于是返回{'flippers':{0:'no',1:'yes'}}
    循环结束，最终得到{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}



'''

mytree = createTree(myDat, labels)
print(mytree)
