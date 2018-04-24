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
    ('GradientBoostingClassifier', GradientBoostingClassifier(learning_rate=0.01, n_estimators=100), 1),
    ('LinearSVC', LinearSVC(C=1.0), 1),
    ('SVC', SVC(C=1.0, degree=3), 1),
    ('DecisionTreeClassifier', DecisionTreeClassifier(criterion='gini'), 2),
    ('ExtraTreeClassifier', ExtraTreeClassifier(criterion='gini'), 2),
    ('RandomForestClassifier', RandomForestClassifier(n_estimators=10, criterion='gini', n_jobs=1), 2),
    ('CalibratedClassifierCV', CalibratedClassifierCV(method='sigmoid', cv=10), 1),
    ('DummyClassifier', DummyClassifier(strategy='stratified'), 2),
    ('AdaBoostClassifier', AdaBoostClassifier(n_estimators=50, learning_rate=0.1, algorithm='SAMME.R'), 1),
    ('BaggingClassifier', BaggingClassifier(n_estimators=10, n_jobs=1), 1),
    ('ExtraTreesClassifier', ExtraTreesClassifier(n_estimators=10, criterion='gini'), 2),
    ('LogisticRegression', LogisticRegression(penalty='l2', max_iter=100, solver='liblinear'), 1),
    ('LogisticRegressionCV', LogisticRegressionCV(Cs=10, penalty='l2', max_iter=100, solver='lbfgs'), 1),
    ('RidgeClassifier', RidgeClassifier(alpha=1.0), 1),
    ('RidgeClassifierCV', RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=10), 2),
    ('SGDClassifier', SGDClassifier(loss='hinge', penalty='l2', epsilon=0.1, n_jobs=1), 1),
    ('BernoulliNB', BernoulliNB(alpha=1.0), 1),
    ('MultinomialNB', MultinomialNB(alpha=1.0), 1),
    ('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='minkowski', n_jobs=1), 2),
    ('RadiusNeighborsClassifier', RadiusNeighborsClassifier(radius=100.0, weights='uniform', algorithm='auto'), 2),
    ('MLPClassifier', MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200), 2),
    ('MLPClassifier', MLPClassifier(hidden_layer_sizes=(256, ), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.0001, max_iter=2000), 2),
    ('MLPClassifier', MLPClassifier(hidden_layer_sizes=(16, 32, 16 ), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200), 2),
    ('MLPClassifier', MLPClassifier(hidden_layer_sizes=(50, 50 ), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200), 2),
    ('MLPClassifier', MLPClassifier(hidden_layer_sizes=(128, 256, 128 ), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200), 2),
    ('LabelPropagation', LabelPropagation(kernel='rbf', n_neighbors=7, max_iter=1000), 1),
]

for i, mod in enumerate(models):
    try:
        trainTarget = trainDF_target if mod[2] == 1 else trainDF_target_dummy
        testTarget = testDF_target if mod[2] == 1 else testDF_target_dummy
        embed()

        ### Step 3) Machine Learning Model
        print("====== Step 3.%s : Fitting Model ======"%(str(i)))
        classModel = mod[1]
        classModel.fit(X = trainDF_feature, y = trainTarget)

        ### Step 4) Calculating Accuracy
        print("====== Step 4.%s : Accuracy Calculation ======"%(str(i)))
        pred = classModel.predict(testDF_feature)
        embed()
        targetWin = [col.argmax() for col in testTarget.as_matrix()]
        predWin = [col.argmax() for col in pred]
        print("Accuracy of %s: %.2f" %(mod[0], accuracy_score(targetWin, predWin)))
    except Exception as ex:
        print(ex)
        print(mod[0], "Failed!")
