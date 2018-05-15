from helper import DataControl, DatasetFormatter, JsonReader, Scorer
import pandas as pd
import numpy as np
from time import time as getTime
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from IPython import embed

lineCount = 20
skipPlots = True
randSeed = 2018
teams = {1: "Team 1", -1: "Team 2"}
heroes = JsonReader('heroes')
lobbies = JsonReader('lobbies')
mods = JsonReader('mods')
regions = JsonReader('regions')
infoObject = {
    'teams': teams,
    'heroes': heroes,
    'lobbies': lobbies,
    'mods': mods,
    'regions': regions
}

print(" ==== Step 1 - Load Dataset ==== ")
columnNames = ['team', 'region', 'mode', 'type']
trainDataset = pd.read_csv('dataset/dota2Train.csv', sep=',', header=None)
for i in range(4):
    trainDataset[i] = trainDataset[i].astype('category')
    trainDataset.rename(columns={i:columnNames[i]}, inplace=True)
print("Train dataset is loaded.")
testDataset = pd.read_csv('dataset/dota2Test.csv', sep=',', header=None)
for i in range(4):
    testDataset[i] = testDataset[i].astype('category')
    testDataset.rename(columns={i:columnNames[i]}, inplace=True)
print("Test dataset is loaded.")
print("-"*lineCount)

print(" ==== Step 2 - Summarize Dataset ==== ")
nTrain = trainDataset.shape[0]
nTest = testDataset.shape[0]
nColumn = trainDataset.shape[1]
print("Number of train instance:\t" + str(nTrain))
print("Number of test instance:\t" + str(nTest))
print("Number of descriptive features:\t" + str(nColumn))
print("-"*lineCount)
print("First 5 row:")
print(trainDataset.head(5))
print("-"*lineCount)
print("Statistics for categorical features:")
print(trainDataset.describe(include='category'))
print("-"*lineCount)
print("Class count of train dataset:")
print(trainDataset.groupby('team').size())
print("-"*lineCount)
print("Class count of test dataset:")
print(testDataset.groupby('team').size())
print("-"*lineCount)
print("Histograms of categorical features:")
categoricDataset = trainDataset.select_dtypes(include='category')
for colName in categoricDataset.columns:
    categoricDataset[colName].value_counts().plot(kind='bar', title=str(colName))
    if not skipPlots:
        plt.show()
print("-"*lineCount)

print(" ==== Step 3 - Example Dota 2 Matches ==== ")
print("TODO HERE")
print("*****************************************************************************************")

print(" ==== Step 4 - Creating Dummies ==== ")
nCategorical = len(trainDataset.select_dtypes(include='category').columns)
trainDataset = pd.get_dummies(trainDataset, columns=list(trainDataset.select_dtypes(include='category').columns))
testDataset = pd.get_dummies(testDataset, columns=list(testDataset.select_dtypes(include='category').columns))
print(str(nCategorical) + " categorical feature found.")
print("Created " + str(trainDataset.shape[1]-nColumn) + " dummy feature created.")
print("New num of column: " + str(trainDataset.shape[1]))
print("-"*lineCount)

print(" ==== Step 5 - Seperating Dataset ==== ")
regexContains = 'team'
regexNotContains = '^((?!team).)*$'
X = trainDataset.filter(regex=regexNotContains)
Y = trainDataset.filter(regex=regexContains)
X_test = testDataset.filter(regex=regexNotContains)
Y_test = testDataset.filter(regex=regexContains)
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=.2, random_state=randSeed)
print("Number of train instance:\t" + str(X_train.shape[0]))
print("Number of validation instance:\t" + str(X_validation.shape[0]))
print("Number of test instance:\t" + str(X_test.shape[0]))
print("-"*lineCount)

print(" ==== Step 6 - Creating Machine Learning Models ==== ")
models = [
    #('Decision Tree', DecisionTreeClassifier(criterion='gini')),
    #('Random Forest', RandomForestClassifier(n_estimators=10, criterion='gini', n_jobs=1)),
    ('K-Nearest Neighbors', KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='minkowski', n_jobs=1)),
    ('MLP tri-layer', MLPClassifier(hidden_layer_sizes=(16, 32, 16 ), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200)),
    #('MLP big-tri-layer', MLPClassifier(hidden_layer_sizes=(128, 256, 128 ), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=200)),
    #('MLP five-layer', MLPClassifier(hidden_layer_sizes=(64, 128, 256, 128, 64 ), activation='relu', solver='adam', alpha=0.0001, learning_rate_init=0.001, max_iter=100)),
]
print("Number of models going to be run:" + str(len(models)))
print("Models:")
for modelName, _ in models:
    print(modelName)
print("-"*lineCount)


print(" ==== Step 7 - Training ==== ")
results = []
for modelname, modelObj in models:
    print(modelname + " training has started")
    start = getTime()
    kfold = model_selection.KFold(n_splits=10, random_state=randSeed)
    scorers = {
        'accr': 'accuracy',
        'prec': 'precision_macro',
        'recl': 'recall_macro'
    }
    scores = model_selection.cross_val_score(modelObj, X_train, Y_train, cv=kfold, scoring=scorers, return_train_score=True)
    results.append(scores)
    embed()
    cv_results = scores['accr']
    print("Results of model " + modelname + ":")
    print("\tMean Accuracy:\t" + str(cv_results.mean()))
    print("\tStd.Dev. Accuracy:\t" + str(cv_results.std()))
    print("\tRun time (in sec.):\t" + str(getTime() - start))

