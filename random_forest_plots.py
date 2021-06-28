import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

# Read the file
path = 'C:/Users/asus/Desktop/imputacja/20.csv'
dataset = pd.read_csv(path, sep=',')
df = pd.DataFrame(dataset)
columns = df.columns.tolist()

# Delete additional useless column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Split data to conditional attributes (X) and decision attribute (y)
X = df.drop(['Choroba'], axis= 1)
y = df['Choroba']
columns = X.columns.tolist()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 2020, stratify=y)

# Check number of trees in the forest
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200, 250, 300, 500]
train_results = []
test_results = []
for estimator in n_estimators:
   rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
line1, = plt.plot(n_estimators, train_results, 'b', label='Dane treningowe')
line2, = plt.plot(n_estimators, test_results, 'r', label='Dane testowe')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Dokładność predykcji')
plt.xlabel('Liczba drzew w lesie')
plt.show()

#Check max number of levels in each decision tree
max_depths = np.linspace(1, 30, 30, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
   rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
line1, = plt.plot(max_depths, train_results, 'b', label='Dane treningowe')
line2, = plt.plot(max_depths, test_results, 'r', label='Dane testowe')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Dokładność predykcji')
plt.xlabel('Głębokość drzewa')
plt.show()

#Check min number of data points placed in a node before the node is split
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   rf = RandomForestClassifier(min_samples_split=min_samples_split)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Dokładność predykcji')
plt.xlabel('Minimalny podział próbek')
plt.show()

#Check min number of data points allowed in a leaf node
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
   rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Dane treningowe')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Dane testujące')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Dokładność predykcji')
plt.xlabel('Minimalne liczba próbek w węźle')
plt.show()

#Check max number of features considered for splitting a node
max_features = list(range(1,len(columns)))
train_results = []
test_results = []
for max_feature in max_features:
   rf = RandomForestClassifier(max_features=max_feature)
   rf.fit(X_train, y_train)
   train_pred = rf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
line1, = plt.plot(max_features, train_results, 'b', label='Dane treningowe')
line2, = plt.plot(max_features, test_results, 'r', label='Dane testujące')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Dokładność predykcji')
plt.xlabel('Maksymalna liczba atrybutów')
plt.show()