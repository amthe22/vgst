# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 15:02:49 2018

@author: cijo
"""

import pandas as pd
import numpy as np
import timeit
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

## preprocessing data
data = pd.read_csv('data.CSV')
required_fields = ['RCONSC', 'SEX', 'AGE', 'RSLEEP', 'RATRIAL', 'RVISINF', 'RSBP', 'RDEF1', 'RDEF2', 'RDEF3',
                   'RDEF4', 'RDEF5', 'RDEF6', 'RDEF7', 'RDEF8', 'STYPE', 'DDIAGISC', 'DDIAGHA', 'DDIAGUN', 'DNOSTRK']
data = data[required_fields]
data.dropna() # dropping fields with NA values or Empty Values

data_d = pd.get_dummies(data)
x = data_d.loc[:, 'AGE':'STYPE_TACS'].values # INPUT DATA READY

## PREPARING OUTPUT ARRAY
data = pd.read_csv('predictstrokedata.csv')
output_data = data[['DDIAGISC', 'DDIAGHA', 'DDIAGUN', 'DNOSTRK']] # REQUIRED FIELDS
output_data = output_data.values
stroke_type=[] # empty array to store final values
for row in output_data:
    row = list(row)
    try:
        stroke_type.append(row.index('Y'))
    except:
        stroke_type.append(-1)
y = stroke_type

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=2)

#classifiers
classifiers = [
    ("Nearest Neighbors", KNeighborsClassifier(3)),
    ("Linear SVM", SVC(kernel="linear", C=0.025)),
    ("RBF SVM", SVC(gamma=2, C=1)),
    ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
    ("Random Forest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ("Multi level Perceptron", MLPClassifier(alpha=1)),
    ("AdaBoost", AdaBoostClassifier()),
    ("NaiveBayes", GaussianNB()),
    ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis())
]
#trianing
for classifier, clf in classifiers:
    print("-" * 5)
    print ("starting training ", classifier)
    start_time = timeit.default_timer()
    model = clf.fit(x_train, y_train)
    print ("training completed in %s seconds"  % (timeit.default_timer() - start_time))

    #testing
    print("starting testing")
    start_time = timeit.default_timer()
    print("Accuracy: %s percent" %(model.score(x_test, y_test) * 100))
    print("testing completed in %s seconds"  % (timeit.default_timer() - start_time))