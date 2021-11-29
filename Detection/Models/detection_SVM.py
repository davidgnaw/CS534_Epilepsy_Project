# Created by: Daniela Chanci Arrubla
# Description: Application of SVM to the features extracted from each patient of the
# publicly available dataset "UPenn and Mayo Clinic's Seizure Detection Challenge" downloaded
# from kaggle

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Initialize list to print
toprint = []

# Load the extracted features
data = pd.read_csv(r"F:\Users\user\Desktop\EMORY\Classes\Fall_2021\CS_534\Project\Detection\Extracted_Features\Train\Dog_1_train.csv")
data = shuffle(data)
data = data.reset_index(drop=True)
X = data.drop(data.iloc[:, [-1]], axis=1) # Discard the class
X = StandardScaler().fit_transform(X) # Data Standardization
y = data.iloc[:, [-1]]
feature_names = data.columns[:-1]

# Print
toprint.append("Dog_1 Detection Summary")
toprint.append("1. One-second eeg segments: {}".format(X.shape[0]))
toprint.append("2. Extracted features: {}".format(X.shape[1]))

# Select best regularization parameter
est = LinearSVC(penalty="l1", dual=False)
c_grid = {'C': np.logspace(-3, 1, num=30)}
clf = GridSearchCV(estimator=est, param_grid=c_grid).fit(X, np.array(y))
reg_param = clf.best_params_['C']

# Feature selection
lsvc = LinearSVC(C=reg_param, penalty="l1", dual=False).fit(X, np.array(y))
model = SelectFromModel(lsvc, prefit=True)
X_reduced = model.transform(X)

# Print
toprint.append("3. Selected features: {}".format(X_reduced.shape[1]))
toprint.append("4. Selected features names:")
toprint.append(list(feature_names[model.get_support()]))

# Balance classes
ros = RandomOverSampler(random_state=42)
X_bal, y_bal = ros.fit_resample(X, np.array(y))
X_bal = StandardScaler().fit_transform(X_bal)

# Print
toprint.append("5. Classes distribution before balancing:")
toprint.append(y.value_counts())
toprint.append("6. Classes distribution after balancing:")
toprint.append(Counter(y_bal))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

# Train SVM
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

# Evaluate model
y_hat = svm_model.predict(X_test)
print(y_hat)

# Write file
for elem in toprint:
    print(elem)