#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by iFantastic on 2019/3/3
# __author__ = 'jun'
import DecisionTree.trees
import DecisionTree.treePlotter

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 年龄，近视远视，散光，
lensesTree = DecisionTree.trees.createTree(lenses, lensesLabels)
print(lensesTree)
DecisionTree.treePlotter.createPlot(lensesTree)
