#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:35:42 2019

@author: balloon_n
"""

import os
os.chdir("/Users/balloon_n/Documents/F/study/class/Fundamental of Data Science/hw3")
import pandas as pd


training = pd.read_csv('training.csv', sep=',')
xTrain = training.iloc[:,:13]
yTrain = training.iloc[:,-1]

testing = pd.read_csv('testing.csv', sep=',')
xTest = testing.iloc[:,:13]
yTest = testing.iloc[:,-1]

#KNN
## p=2
from sklearn.neighbors import KNeighborsClassifier
resultsKNN1=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=2,metric='minkowski')
    knn.fit(xTrain,yTrain)
    scoreTrain=knn.score(xTrain,yTrain)
    scoreTest=knn.score(xTest,yTest)
    resultsKNN1.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN1.head(11))
resultsKNN1.pop('KNN')
ax1=resultsKNN1.plot()

## p = 1
resultsKNN2=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=1,metric='minkowski')
    knn.fit(xTrain,yTrain)
    scoreTrain=knn.score(xTrain,yTrain)
    scoreTest=knn.score(xTest,yTest)
    resultsKNN2.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN2.head(11))
resultsKNN2.pop('KNN')
ax2=resultsKNN2.plot()

## p=2 with weight
resultsKNN3=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=2,metric='minkowski',weights="distance")
    knn.fit(xTrain,yTrain)
    scoreTrain=knn.score(xTrain,yTrain)
    scoreTest=knn.score(xTest,yTest)
    resultsKNN3.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN3.head(11))
resultsKNN3.pop('KNN')
ax3=resultsKNN3.plot()

## p=1 with weight
resultsKNN4=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=1,metric='minkowski',weights="distance")
    knn.fit(xTrain,yTrain)
    scoreTrain=knn.score(xTrain,yTrain)
    scoreTest=knn.score(xTest,yTest)
    resultsKNN4.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN4.head(11))
resultsKNN4.pop('KNN')
ax4=resultsKNN4.plot()

