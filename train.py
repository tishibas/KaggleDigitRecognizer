#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import cv
import sys
import numpy as np

from sklearn import cross_validation
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing

X = np.loadtxt("train_20x20_normalized.txt")
Y = np.loadtxt("label.txt")

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
X_train_minmax = min_max_scaler.fit_transform(X)
np.savetxt("scale_normal_sq.txt", min_max_scaler.scale_)

tuned_parameters = [
        {'C': [1,10,100,1000], 'gamma': [0.1,0.01,0.001,0.0001], 'kernel': ['rbf']},
        ]
clf = GridSearchCV(svm.SVC(),tuned_parameters,n_jobs=3)
clf.fit(X_train_minmax,Y,cv=5)
_ = joblib.dump(clf, "model.pkl", compress=3)
print clf.best_estimator_
print clf.best_score_
print clf.grid_scores_
sys.exit()

