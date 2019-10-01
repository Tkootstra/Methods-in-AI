# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:11:06 2019

@author: Tim.O
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import pickle
df = pd.read_csv('dfAllFeatures.csv')

X = df.drop('label', axis=1)
y = df['label']
#y = LabelBinarizer().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.15, random_state=101)

#folder = StratifiedKFold(n_splits=10)
#
#for trainIdx, testIdx in folder.split(X_train, y_train):
#    X_trainCV, X_testCV = X_train[trainIdx], X_train[testIdx]
#    y_trainCV, y_testCV = y_train[trainIdx], y_train[testIdx]
#    
#
#
model = RandomForestClassifier(n_estimators=10)

parameters = {'max_depth':[20,50], 'max_features':[1, 10],\
              'n_estimators':[200,300,500], 'criterion': ('gini', 'entropy')}

gridSearchPipe = GridSearchCV(model, parameters, cv=5, scoring='f1_weighted', n_jobs=15)

gridSearchPipe.fit(X, y)
bestParams = list(gridSearchPipe.best_params_.values())

bestModel = RandomForestClassifier(criterion=bestParams[0], max_depth=bestParams[1],\
                                   max_features=bestParams[2], n_estimators=bestParams[3])

bestModel.fit(X_train, y_train)
filename = 'finalized_model.sav'
pickle.dump(bestModel, open(filename, 'wb'))