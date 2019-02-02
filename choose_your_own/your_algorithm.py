#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn import neighbors
from sklearn import ensemble
from sklearn import tree

#clf = neighbors.KNeighborsClassifier(n_neighbors=50)
#clf = tree.DecisionTreeClassifier(min_samples_split=70)
#clf = ensemble.AdaBoostClassifier(n_estimators=70)
clf = ensemble.RandomForestClassifier(min_samples_split=70, n_estimators=70)

clf.fit(features_train, labels_train)

from sklearn import metrics
predictions = clf.predict(features_test)
acc = metrics.accuracy_score(labels_test, predictions)
print "Accuracy:", acc
print "Number of training examples:", len(features_train)





try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
