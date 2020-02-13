#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:12:12 2019

@author: balloon_n
"""

import os
os.chdir("/Users/balloon_n/Documents/F/study/class/Fundamental of Data Science/hw3")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

NormalizedDF= pd.read_csv('wineNormalized.csv', sep=',') 
NormalizedDFnew= NormalizedDF.drop(columns=['Proanthocyanins', 'Malic acid'])

y = NormalizedDFnew.pop("Class")
 
# derive new features based on correlation 
correlation = NormalizedDFnew.corr()
plt.figure(figsize=(10,8))
mask = np.zeros_like(correlation)#https://seaborn.pydata.org/generated/seaborn.heatmap.html
mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation,linewidths=.3,annot=True,mask=mask,cmap="YlGnBu",cbar=False)
NormalizedDFnew.columns

wine=NormalizedDFnew.iloc[:,:11]
Newwine=wine.copy()
Newwine['Flavanoids+Total phenols']=Newwine['Flavanoids']+Newwine['Total phenols']
Newwine['Proline+Alcohol']=Newwine['Proline']+Newwine['Alcohol']
#Newwine['Color intensity+Alcohol']=Newwine['Color intensity']+Newwine['Alcohol']
#Newwine['Alcohol+Proline']=Newwine['Alcohol']+Newwine['Proline']
#Newwine['Color intensity+Flavanoids']=Newwine['Color intensity']+Newwine['Flavanoids']
#Newwine['Ash+Alcalinity of ash']=Newwine['Ash']+Newwine['Alcalinity of ash']

print(Newwine.head(11))

##training testing split
xnew = Newwine
xnewTrain, xnewTest, yTrain, yTest = train_test_split(xnew, y, test_size = 1/3,random_state =1,stratify=y)

##test new dataset with derived descriptive features.
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

### Entropy
resultsEntropy = pd.DataFrame(columns=['LevelLimit', 'Score for Training', 'Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion='entropy', max_depth=treeDepth, random_state=0)
    dct= dct.fit(xnewTrain,yTrain)
    dct.predict(xnewTest)
    scoreTrain = dct.score(xnewTrain,yTrain)
    scoreTest = dct.score(xnewTest,yTest)
    resultsEntropy.loc[treeDepth] = [treeDepth, scoreTrain, scoreTest]
print(resultsEntropy.head(11))
resultsEntropy.pop('LevelLimit')
ax1 = resultsEntropy.plot()

### Gini
resultsGini = pd.DataFrame(columns=['LevelLimit', 'Score for Training', 'Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion='gini', max_depth=treeDepth, random_state=0)
    dct= dct.fit(xnewTrain,yTrain)
    dct.predict(xnewTest)
    scoreTrain = dct.score(xnewTrain,yTrain)
    scoreTest = dct.score(xnewTest,yTest)
    resultsGini.loc[treeDepth] = [treeDepth, scoreTrain, scoreTest]
print(resultsGini.head(11))
resultsGini.pop('LevelLimit')
ax2 = resultsGini.plot()

### p = 2
resultsKNN1=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=2,metric='minkowski')
    knn.fit(xnewTrain,yTrain)
    scoreTrain=knn.score(xnewTrain,yTrain)
    scoreTest=knn.score(xnewTest,yTest)
    resultsKNN1.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN1.head(11))
resultsKNN1.pop('KNN')
axKNN1=resultsKNN1.plot()

### p = 1
resultsKNN2=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=1,metric='minkowski')
    knn.fit(xnewTrain,yTrain)
    scoreTrain=knn.score(xnewTrain,yTrain)
    scoreTest=knn.score(xnewTest,yTest)
    resultsKNN2.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN2.head(11))
resultsKNN2.pop('KNN')
axKNN2=resultsKNN2.plot()

### p=2 with weight
resultsKNN3=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=2,metric='minkowski',weights="distance")
    knn.fit(xnewTrain,yTrain)
    scoreTrain=knn.score(xnewTrain,yTrain)
    scoreTest=knn.score(xnewTest,yTest)
    resultsKNN3.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN3.head(11))
resultsKNN3.pop('KNN')
axKNN3=resultsKNN3.plot()

### p=1 with weight
resultsKNN4=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=1,metric='minkowski',weights="distance")
    knn.fit(xnewTrain,yTrain)
    scoreTrain=knn.score(xnewTrain,yTrain)
    scoreTest=knn.score(xnewTest,yTest)
    resultsKNN4.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN4.head(11))
resultsKNN4.pop('KNN')
axKNN4=resultsKNN4.plot()

### random forest
randomforest=pd.DataFrame(columns=['RandomForest','Score for Training','Score for Testing'])
for n in range(1,20):
    feat_labels = xnew.columns[:]
    forest=RandomForestClassifier(n_estimators=n,random_state =1,n_jobs=-1)
    forest.fit(xnewTrain,yTrain)
    score1=forest.score(xnewTrain,yTrain)
    score2=forest.score(xnewTest,yTest)
    randomforest.loc[n]=[n,score1,score2]
print(randomforest.head(20))
randomforest.pop('RandomForest')
ax=randomforest.plot()

importance=forest.feature_importances_
indices = np.argsort(importance)[::-1]
for i in range(xnewTrain.shape[1]):
    print("(%2d) %-*s %f" % (i+1,30, feat_labels[indices[i]], importance[indices[i]]))
    
    
#feature selection /  drop features
xnewTrain1=xnewTrain.drop(columns=['Ash','Proline','Alcohol','Flavanoids','Total phenols'])
xnewTest1=xnewTest.drop(columns=['Ash','Proline','Alcohol','Flavanoids','Total phenols'])
xnewTrain1.columns

### Entropy
resultsEntropy = pd.DataFrame(columns=['LevelLimit', 'Score for Training', 'Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion='entropy', max_depth=treeDepth, random_state=0)
    dct= dct.fit(xnewTrain1,yTrain)
    dct.predict(xnewTest1)
    scoreTrain = dct.score(xnewTrain1,yTrain)
    scoreTest = dct.score(xnewTest1,yTest)
    resultsEntropy.loc[treeDepth] = [treeDepth, scoreTrain, scoreTest]
print(resultsEntropy.head(11))
resultsEntropy.pop('LevelLimit')
ax1 = resultsEntropy.plot()

### Gini
resultsGini = pd.DataFrame(columns=['LevelLimit', 'Score for Training', 'Score for Testing'])
for treeDepth in range (1,11):
    dct = DecisionTreeClassifier(criterion='gini', max_depth=treeDepth, random_state=0)
    dct= dct.fit(xnewTrain1,yTrain)
    dct.predict(xnewTest1)
    scoreTrain = dct.score(xnewTrain1,yTrain)
    scoreTest = dct.score(xnewTest1,yTest)
    resultsGini.loc[treeDepth] = [treeDepth, scoreTrain, scoreTest]
print(resultsGini.head(11))
resultsGini.pop('LevelLimit')
ax2 = resultsGini.plot()

### p = 2
resultsKNN1=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=2,metric='minkowski')
    knn.fit(xnewTrain1,yTrain)
    scoreTrain=knn.score(xnewTrain1,yTrain)
    scoreTest=knn.score(xnewTest1,yTest)
    resultsKNN1.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN1.head(11))
resultsKNN1.pop('KNN')
axKNN1=resultsKNN1.plot()

### p = 1
resultsKNN2=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=1,metric='minkowski')
    knn.fit(xnewTrain1,yTrain)
    scoreTrain=knn.score(xnewTrain1,yTrain)
    scoreTest=knn.score(xnewTest1,yTest)
    resultsKNN2.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN2.head(11))
resultsKNN2.pop('KNN')
axKNN2=resultsKNN2.plot()

### p=2 with weight
resultsKNN3=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=2,metric='minkowski',weights="distance")
    knn.fit(xnewTrain1,yTrain)
    scoreTrain=knn.score(xnewTrain1,yTrain)
    scoreTest=knn.score(xnewTest1,yTest)
    resultsKNN3.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN3.head(11))
resultsKNN3.pop('KNN')
axKNN3=resultsKNN3.plot()

### p=1 with weight
resultsKNN4=pd.DataFrame(columns=['KNN','Score for Training','Score for Testing'])
for knncount in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=knncount,p=1,metric='minkowski',weights="distance")
    knn.fit(xnewTrain1,yTrain)
    scoreTrain=knn.score(xnewTrain1,yTrain)
    scoreTest=knn.score(xnewTest1,yTest)
    resultsKNN4.loc[knncount]=[knncount,scoreTrain,scoreTest]
print(resultsKNN4.head(11))
resultsKNN4.pop('KNN')
axKNN4=resultsKNN4.plot()

### random forest
randomforest=pd.DataFrame(columns=['RandomForest','Score for Training','Score for Testing'])
for n in range(1,20):
    feat_labels = xnewTrain1.columns[:]
    forest=RandomForestClassifier(n_estimators=n,random_state =1,n_jobs=-1)
    forest.fit(xnewTrain1,yTrain)
    score1=forest.score(xnewTrain1,yTrain)
    score2=forest.score(xnewTest1,yTest)
    randomforest.loc[n]=[n,score1,score2]
print(randomforest.head(20))
randomforest.pop('RandomForest')
ax=randomforest.plot()

importance=forest.feature_importances_
indices = np.argsort(importance)[::-1]
for i in range(xnewTrain1.shape[1]):
    print("(%2d) %-*s %f" % (i+1,30, feat_labels[indices[i]], importance[indices[i]]))
    
    