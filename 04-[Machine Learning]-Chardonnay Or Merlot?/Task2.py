#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:20:00 2019

@author: balloon_n
"""

import os
os.chdir("/Users/balloon_n/Documents/F/study/class/Fundamental of Data Science/hw3")
import pandas as pd

NormalizedDF = pd.read_csv('wineNormalized.csv', sep=',')
print(NormalizedDF.head())

#stratified holdout sampling and save file.
x = NormalizedDF.iloc[:,:13]
y = NormalizedDF.iloc[:,-1]
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 1,stratify=y)

training=pd.concat([xTrain,yTrain],axis=1)
testing=pd.concat([xTest,yTest],axis=1)
training.to_csv('training.csv', index = False )
testing.to_csv('testing.csv', index = False )

#decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

##entropy
resultsEntropy = pd.DataFrame(columns=['LevelLimit', 'Score for Training', 'Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion='entropy', max_depth=treeDepth, random_state=0)
    dct= dct.fit(xTrain,yTrain)
    dct.predict(xTest)
    scoreTrain = dct.score(xTrain,yTrain)
    scoreTest = dct.score(xTest,yTest)
    resultsEntropy.loc[treeDepth] = [treeDepth, scoreTrain, scoreTest]
print(resultsEntropy.head(11))
resultsEntropy.pop('LevelLimit')
ax1 = resultsEntropy.plot()

dct1 = DecisionTreeClassifier(criterion = "entropy", max_depth = 2, random_state = 0)
dct1 = dct1.fit(xTrain,yTrain)
export_graphviz(dct1, out_file = "entropytree.dot")

##Gini
resultsGini = pd.DataFrame(columns=['LevelLimit', 'Score for Training', 'Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion='gini', max_depth=treeDepth, random_state=0)
    dct= dct.fit(xTrain,yTrain)
    dct.predict(xTest)
    scoreTrain = dct.score(xTrain,yTrain)
    scoreTest = dct.score(xTest,yTest)
    resultsGini.loc[treeDepth] = [treeDepth, scoreTrain, scoreTest]
print(resultsGini.head(11))
resultsGini.pop('LevelLimit')
ax2 = resultsGini.plot()

dct2 = DecisionTreeClassifier(criterion = "gini", max_depth = 2, random_state = 0)
dct2 = dct2.fit(xTrain,yTrain)
export_graphviz(dct2, out_file = "ginitree.dot")