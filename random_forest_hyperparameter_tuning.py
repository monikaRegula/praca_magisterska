import statistics
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

def print_accuracy(accuracy_score, score_text=False):
    """
    Take an accuracy score between 0 and 1 and print output to screen cleanly
    """
    clean_accuracy = accuracy_score*100.0
    if score_text:
        clean_text = score_text.strip() + ' '
        print('{}{:.2f}%'.format(clean_text, clean_accuracy))
    else:
        print('{:.2f}%'.format(clean_accuracy))

def generate_noise(rows, percentage):
    mean = statistics.mean(rows)
    lim = mean * percentage
    random_noise = np.random.uniform(low=-lim, high=lim, size=(len(rows)))
    noise = np.random.normal(lim, 0.1, len(rows))
    noised_data = []
    for i in range(0,len(random_noise)):
        new = noise[i] + rows[i]
        noised_data.append(round(new,2))

    return noised_data

def add_noise_to_dataset(dataset, percentage):
    columns = dataset.columns.tolist()
    newColumns = {}
    for column in columns:
        rows = dataset[column].tolist()
        newColumns[column] = generate_noise(rows, percentage)
    return pd.DataFrame(newColumns)

# Read the csv file
path = 'C:/Users/asus/Desktop/imputacja/50.csv'
dataset = pd.read_csv(path, sep=',')
df = pd.DataFrame(dataset)
columns = df.columns.tolist()

# Delete additional useless column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Split data to conditional attributes (X) and decision attribute (y)
X = df.drop(['Choroba'], axis= 1)
y = df["Choroba"]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 2020, stratify=y)

# Optional addition of noise to dataset (check of model's resistance)
# X = add_noise_to_dataset(X, 0.03)

rf = RandomForestClassifier(max_depth=10, max_features='sqrt', bootstrap=False, min_samples_leaf=0.2, n_estimators=500)
rf2 = RandomForestClassifier(bootstrap= False, max_depth= 11, max_features= 'sqrt', min_samples_leaf= 0.1, min_samples_split= 0.2, n_estimators= 500)
rf3 = RandomForestClassifier(bootstrap= False, max_depth= 11, max_features= 'sqrt', min_samples_leaf= 0.1, min_samples_split= 0.2, n_estimators= 200)
rf4 = RandomForestClassifier(bootstrap= False, max_depth= 11, max_features= 'sqrt', min_samples_leaf= 2, min_samples_split= 3, n_estimators= 500)
rf5 = RandomForestClassifier(bootstrap= False, max_depth= 11, max_features= 'sqrt', min_samples_leaf= 0.1, min_samples_split= 0.2, n_estimators= 200)

models = []
models.append(("RF",rf))
models.append(("RF2",rf2))
models.append(("RF3",rf3))
models.append(("RF4",rf4))
models.append(("RF5",rf5))

results = []
names = []
seed = 10
scoring = 'accuracy'

# Evaluating estimator performance
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Predictions on validation dataset
for name,model in models:
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print(name ," Accuracy: ",accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

# Voting Ensemble for Classification
ensemble = VotingClassifier(models)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
print(ensemble.__class__.__name__, accuracy_score(y_test, y_pred))


# Compare algorithms
fig = plt.figure(figsize=(70,20))
fig.suptitle('Porównanie algorytmów klasyfikacji' ,fontsize=50)
ax = fig.add_subplot(111)
plt.ylabel('Dokładność predykcji [%]', fontsize=50)
plt.boxplot(results)
plt.yticks(fontsize=30)
ax.set_xticklabels(names, fontsize=30)
plt.show()

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print(rfc.score(X_train, y_train))

rfc2 = RandomForestClassifier(n_estimators=10)
rfc2.fit(X_train, y_train)
print("ACCURACY: ", rfc2.score(X_train, y_train))
print("=====================================================")

# Feature importance for random forest
feats = {}
for feature, importance in zip(X.columns, rfc.feature_importances_):
    feats[feature] = importance
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
importances = importances.sort_values(by='Gini-Importance', ascending=False)
importances = importances.reset_index()
importances = importances.rename(columns={'index': 'Features'})
sns.set(font_scale = 5)
sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
fig, ax = plt.subplots()
fig.set_size_inches(30,15)
sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
plt.xlabel('Istotność', fontsize=25, weight = 'bold')
plt.ylabel('Atrybuty', fontsize=25, weight = 'bold')
plt.title('Istotność atrybutów', fontsize=25, weight = 'bold')
plt.show()
print(importances)


# Hyperparameter Tuning Round 1: RandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
rfc_2 = RandomForestClassifier()
rfc_2.fit(X_train, y_train)
print(rfc_2.score(X_train, y_train))
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(start = 1, stop = 30, num = 30)]
min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True)
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
bootstrap = [True, False]
param_dist = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rs = RandomizedSearchCV(rfc_2,
                        param_dist,
                        n_iter = 100,
                        cv = 3,
                        verbose = 1,
                        n_jobs=-1,
                        random_state=0)
grid_result = rs.fit(X_train, y_train)
print("RS PARAMS: ", rs.best_params_)
print_accuracy(accuracy_score(y_test, grid_result.best_estimator_.predict(X_test)), 'Random Forest Classifier with Random Search:')
print("============================================================")

# Hyperparameter Tuning Round 2: GridSearchCV
n_estimators = [100,200,300,500]
max_features = ['sqrt']
max_depth = [2,3,7,10,11,15]
min_samples_split = np.linspace(0.1, 1.0, 10, endpoint=True)
min_samples_leaf = np.linspace(0.1, 0.5, 5, endpoint=True)
bootstrap = [False]
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
gs = GridSearchCV(rfc, param_grid, cv = 3, verbose = 1, n_jobs=-1)
grid_result = gs.fit(X_train, y_train)
rfc_3 = gs.best_estimator_
print("GS PARAMS: ", gs.best_params_)
print_accuracy(accuracy_score(y_test, grid_result.best_estimator_.predict(X_test)), 'Random Forest Classifier with Grid Search:')