from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

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
n = len(columns)-1

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,
    stratify=y, random_state=0)

# Standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Computation the coviarance matrix of the standardized training dataset
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# Calculate cumulative sum of explained variances
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Plot explained variances
plt.bar(range(1,n), var_exp, alpha=0.5,
        align='center', label='Pojedyncza wariancja')
plt.step(range(1,n), cum_var_exp, where='mid',
         label='Łączna wariancja')
plt.ylabel('Współczynnik wariancji')
plt.xlabel('Indeks głównej składowej')
plt.legend(loc='best')
plt.show()