#!/user/bin/env python
#coding=utf-8

import matplotlib.pyplot as plt

#定义文本框和箭头格式,常量
decisionNode = dict(boxstyle = "sawtooth", fc="0.8")
leafNode = dict(boxstyle = "round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):#执行绘图功能
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center",bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor="white")#创建一个新图形
    fig.clf()#清空图形绘图区
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)#从第一个点到第二个点
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

createPlot()