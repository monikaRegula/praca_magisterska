import statistics
from matplotlib import pyplot
from sklearn import model_selection
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt

# Read the csv file
# path = 'C:/Users/asus/Desktop/imputacja/20.csv'
path = 'C:/Users/asus/Desktop/imputacja/50.csv'
dataset = pd.read_csv(path, sep=',')
df = pd.DataFrame(dataset)

# Delete additional useless column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Split data to conditional attributes (X) and decision attribute (y)
X = df.drop(['Choroba'], axis= 1)
attributes_names = X.columns.tolist()
y = df["Choroba"]

# Print shapes of attributes
print(f"'X' shape: {X.shape}")
print(f"'y' shape: {y.shape}")

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

# Optional addition of noise to dataset (check of model's resistance)
# X = add_noise_to_dataset(X, 0.03)

# Split dataset into training set and test set
X_train, X_test, Y_train, y_test = train_test_split(X,y,test_size=0.3, shuffle=True, random_state=10)
seed=10
scoring = 'accuracy'

# Dimensional reduction with PCA
# n_components = 20
n_components = 7
pca = PCA(n_components=n_components)

#Attributes scaling
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
y_train = np.array(Y_train)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

lr = LogisticRegression(solver='newton-cg',penalty='l2',C=100)
lr2 = LogisticRegression(solver='lbfgs',penalty='l2',C=100)
lr3 = LogisticRegression(solver='liblinear',penalty='l2',C=100)
lr4 = LogisticRegression(solver='sag',penalty='l2',C=100)
lr5 = LogisticRegression(solver='saga',penalty='l2',C=100)
lr6 = LogisticRegression(solver='liblinear',penalty='l1',C=100)
lr7 = LogisticRegression(solver='saga',penalty='l1',C=100)
lr8 = LogisticRegression(solver='liblinear',penalty='l1',C=10)
lr9 = LogisticRegression(solver='saga',penalty='l1',C=10)
lr9 = LogisticRegression(solver='liblinear',penalty='l1',C=1)
lr10 = LogisticRegression(solver='saga',penalty='l1',C=1)
lr11 = LogisticRegression(solver='liblinear',penalty='l2',C=0.1)
lr12 = LogisticRegression(solver='saga',penalty='l1',C=0.1)
lr13 = LogisticRegression(solver='liblinear',penalty='l2',C=0.01)
lr14 = LogisticRegression(solver='saga',penalty='l2',C=0.01)
lr15 = LogisticRegression(solver='newton-cg',penalty='l2',C=0.1)
lr16 = LogisticRegression(solver='newton-cg',penalty='l2',C=10)

knn = KNeighborsClassifier(n_neighbors=9,metric='euclidean',weights='distance')
knn2 = KNeighborsClassifier(n_neighbors=9,metric='euclidean',weights='uniform')
knn3 = KNeighborsClassifier(n_neighbors=30,metric='euclidean',weights='distance')
knn4 = KNeighborsClassifier(n_neighbors=30,metric='euclidean',weights='uniform')
knn5 = KNeighborsClassifier(n_neighbors=5,metric='euclidean',weights='distance')
knn6 = KNeighborsClassifier(n_neighbors=5,metric='euclidean',weights='uniform')
knn7 = KNeighborsClassifier(n_neighbors=13,metric='euclidean',weights='distance')
knn8 = KNeighborsClassifier(n_neighbors=13,metric='euclidean',weights='uniform')
knn9 = KNeighborsClassifier(n_neighbors=9,metric='minkowski',weights='distance')
knn10 = KNeighborsClassifier(n_neighbors=9,metric='minkowski',weights='uniform')
knn11 = KNeighborsClassifier(n_neighbors=11,metric='minkowski',weights='distance')
knn12 = KNeighborsClassifier(n_neighbors=11,metric='minkowski',weights='uniform')
knn13 = KNeighborsClassifier(n_neighbors=35,metric='manhattan',weights='distance')

svc = SVC(C=0.1, gamma=0.01, kernel='rbf')
svc2 = SVC(C=10, gamma=0.01, kernel='rbf')
svc3 = SVC(C=50, gamma=0.01, kernel='rbf')
svc4 = SVC(C=1000, gamma=0.001, kernel='rbf')
svc5 = SVC(C=50, gamma=0.001, kernel='sigmoid')
svc6 = SVC(C=1000, gamma=0.001, kernel='sigmoid')
svc7 = SVC(C=25,  kernel='linear')
svc8 = SVC(C=50,  kernel='linear')
svc9 = SVC(C=100,  kernel='linear')
svc10 = SVC(C=1000,  kernel='linear')
svc11 = SVC(degree= 1, kernel= 'poly')

models = []
models.append(("LR", lr))
models.append(("LR2", lr2))
models.append(("LR3", lr3))
models.append(("LR4", lr4))
models.append(("LR5", lr5))
models.append(("LR6", lr6))
models.append(("LR7", lr7))
models.append(("LR8", lr8))
models.append(("LR9", lr9))
models.append(("LR10", lr10))
models.append(("LR11", lr11))
models.append(("LR12", lr12))
models.append(("LR13", lr13))
models.append(("LR14", lr14))
models.append(("LR15", lr15))
models.append(("LR16", lr16))

models.append(("KNN", knn))
models.append(("KNN2", knn2))
models.append(("KNN3", knn3))
models.append(("KNN4", knn4))
models.append(("KNN5", knn5))
models.append(("KNN6", knn6))
models.append(("KNN7", knn7))
models.append(("KNN8", knn8))
models.append(("KNN9", knn9))
models.append(("KNN10", knn10))
models.append(("KNN11", knn11))
models.append(("KNN12", knn12))
models.append(("KNN13", knn13))

models.append(("SVC",svc))
models.append(("SVC2",svc2))
models.append(("SVC3",svc3))
models.append(("SVC4",svc4))
models.append(("SVC5",svc5))
models.append(("SVC6",svc6))
models.append(("SVC7",svc7))
models.append(("SVC8",svc8))
models.append(("SVC9",svc9))
models.append(("SVC10",svc10))
models.append(("SVC11",svc11))

results = []
names = []

# Evaluating estimator performance
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed,shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Comparison of algorithms
fig = plt.figure(figsize=(70,20))
fig.suptitle('Porównanie algorytmów klasyfikacji n_components =  '+str(n_components) ,fontsize=50)
ax = fig.add_subplot(111)
plt.ylabel('Dokładność predykcji [%]', fontsize=50)
plt.boxplot(results)
plt.yticks(fontsize=30)
ax.set_xticklabels(names, fontsize=30)
plt.show()

# Predictions on validation dataset
for name,model in models:
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    print(name ," Accuracy: ",accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    # Feature importance for Logistic Regression
    if 'LR' in name:
        # get importance
        importance = model.coef_[0]
        # summarize feature importance
        for i, v in enumerate(importance):
            attribute = attributes_names[i]
            print(attribute, 'Atrybut: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.title(name)
        pyplot.show()
    print("==================================================================")

# Voting Ensemble for Classification
from sklearn.ensemble import VotingClassifier
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
# Create the ensemble model
ensemble = VotingClassifier(models)
results = model_selection.cross_val_score(ensemble, X_train, Y_train, cv=kfold)
print("VotingClassifier: ", results.mean())

# Feature importance for best SVC model
svc4.fit(X_train, y_train)
perm_importance = permutation_importance(svc4, X_test, y_test)
features = np.array(attributes_names)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Algorytm ważności permutacji")
plt.show()