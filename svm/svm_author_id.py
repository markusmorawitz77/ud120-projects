#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC

features_train = features_train[:len(features_train)/1] 
labels_train = labels_train[:len(labels_train)/1] 

for c in [10000]:
    print 
    print "train with C =", c
    clf = SVC(kernel="rbf", gamma="auto", C=c)
    t0 = time()
    clf.fit(features_train, labels_train)
    print "traning time:", round(time()-t0, 3), " s"

    print "number of test data sets:", len(features_test)
    t0 = time()
    accuracy = clf.score(features_test, labels_test)
    print "test time:", round(time()-t0, 3), " s"
    print "accuracy:", accuracy

    p = clf.predict(features_test)
    zeros = 0
    ones = 0
    for y in p:
        if y == 1: 
            ones += 1
        elif y == 0: 
            zeros += 1
    print "0:", zeros
    print "1:", ones


#########################################################


