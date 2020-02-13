#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:51:08 2019

@author: balloon_n
"""

import os
os.chdir("/Users/balloon_n/Documents/F/study/class/Fundamental of Data Science/hw3")
import pandas as pd

#reading a file
df = pd.read_csv('wineData.csv', sep=',')
print(df.head())
print(df.shape)
#print(df.dtypes)
#print(df.columns)

#identify Class attribute and perform class mapping 
import numpy as np
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['Class']))}
print(class_mapping)
df['Class'] = df['Class'].map(class_mapping)
df['Class'].value_counts().plot(kind='bar',color=tuple(["green", "orange"]))
print(df.head())

#normalize all remaining attributes to [0, 3] range using Min‐Max
##https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
from sklearn import preprocessing
dfTemp = df.iloc[:,1:]
x = df.iloc[:,1:].values
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 3), copy=True)
x_scaled = min_max_scaler.fit_transform(x)
NormalizedDF = pd.DataFrame(x_scaled, columns=dfTemp.columns)
NormalizedDF["Class"] = df["Class"]
print(NormalizedDF.head(11))

#save the entire data set as wineNormalized.csv file
NormalizedDF.to_csv('wineNormalized.csv', index = False )


