import numpy as np
import pandas as pd
from data_reader import getDataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from IPython import embed

### Step 1) Preprocessing Dataset
print("====== Step 1 : Preprocessing ======")
trainData, testData = getDataset()
# Removing all rows that contain nan value
trainData = trainData[~np.isnan(trainData).any(axis=1)]
testData = testData[~np.isnan(testData).any(axis=1)]

### Step 2) Converting to Dataframes and marking categorical
print("====== Step 2 : Converting to Dataframe ======")
trainDF = pd.DataFrame(data=trainData, dtype='int')
testDF = pd.DataFrame(data=testData, dtype='int')

trainDF_target = [trainDF[col].astype('category') for col in trainDF.columns[:1]]
trainDF_target = pd.DataFrame(trainDF_target).T
trainDF_target_dummy = pd.get_dummies(trainDF_target, columns=list(trainDF_target.columns), prefix="bin")
trainDF_feature = [trainDF[col].astype('category') for col in trainDF.columns[1:]]
trainDF_feature = pd.DataFrame(trainDF_feature).T

testDF_target = [testDF[col].astype('category') for col in testDF.columns[:1]]
testDF_target = pd.DataFrame(testDF_target).T
testDF_target_dummy = pd.get_dummies(testDF_target, columns=list(testDF_target.columns), prefix="bin")
testDF_feature = [testDF[col].astype('category') for col in testDF.columns[1:]]
testDF_feature = pd.DataFrame(testDF_feature).T

models = [
    ('DecisionTreeClassifier', DecisionTreeClassifier(criterion='gini'), 2),
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=10, criterion='gini', n_jobs=1), 2),
    ('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='minkowski', n_jobs=1), 2),
    ('MLPClassifier', MLPClassifier(hidden_layer_sizes=(16, 32, 16 ), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200), 2),
    ('MLPClassifier', MLPClassifier(hidden_layer_sizes=(128, 256, 128 ), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200), 2),
    ('MLPClassifier', MLPClassifier(hidden_layer_sizes=(64, 128, 256, 128, 64 ), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=100), 2),
]

for i, mod in enumerate(models):
    try:
        trainTarget = trainDF_target_dummy
        testTarget = testDF_target_dummy

        ### Step 3) Machine Learning Model
        print("====== Step 3.%s : Fitting %s Model ======"%(str(i), mod[0]))
        classModel = mod[1]
        classModel.fit(X = trainDF_feature, y = trainTarget)

        ### Step 4) Calculating Accuracy
        print("====== Step 4.%s : Accuracy Calculation ======"%(str(i)))
        pred = classModel.predict(testDF_feature)
        targetWin = [col.argmax() for col in testTarget.as_matrix()]
        predWin = [col.argmax() for col in pred]
        print("Accuracy of %s: %.2f" %(mod[0], accuracy_score(targetWin, predWin)))
    except Exception as ex:
        print(ex)
        print(mod[0], "Failed!")
