#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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
from sklearn import tree
from sklearn import metrics

print "Number of features:", len(features_train[0])
clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print "traning time:", round(time()-t0, 3), " s"

print "number of test data sets:", len(features_test)
t0 = time()
predictions = clf.predict(features_test)
accuracy = metrics.accuracy_score(labels_test, predictions)
print "test time:", round(time()-t0, 3), " s"
print "accuracy:", accuracy

#########################################################


