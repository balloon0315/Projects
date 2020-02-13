#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 10:40:38 2019

@author: balloon_n
"""


import os
os.chdir("/Users/balloon_n/Documents/F/study/class/Fundamental of Data Science/hw3")
import pandas as pd
import numpy as np

training = pd.read_csv('training.csv', sep=',')
xTrain = training.iloc[:,:13]
yTrain = training.iloc[:,-1]

testing = pd.read_csv('testing.csv', sep=',')
xTest = testing.iloc[:,:13]
yTest = testing.iloc[:,-1]

from sklearn.ensemble import RandomForestClassifier
randomforest=pd.DataFrame(columns=['RandomForest','Score for Training','Score for Testing'])
for n in range(1,20):
    feat_labels = x.columns[:]
    forest=RandomForestClassifier(n_estimators=n,random_state =1,n_jobs=-1)
    forest.fit(xTrain,yTrain)
    score1=forest.score(xTrain,yTrain)
    score2=forest.score(xTest,yTest)
    randomforest.loc[n]=[n,score1,score2]
print(randomforest.head(20))
randomforest.pop('RandomForest')
ax=randomforest.plot()

importance=forest.feature_importances_
indices = np.argsort(importance)[::-1]
for i in range(xTrain.shape[1]):
    print("(%2d) %-*s %f" % (i+1,30, feat_labels[indices[i]], importance[indices[i]]))
    

   
