# Created by: Daniela Chanci Arrubla
# Description: Application of SVM to the features extracted from each patient of the
# publicly available dataset "UPenn and Mayo Clinic's Seizure Detection Challenge" downloaded
# from kaggle

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the extracted features
data = pd.read_csv(r"F:\Users\user\Desktop\EMORY\Classes\Fall_2021\CS_534\Project\Detection\Extracted_Features\Train\Dog_1_train.csv")
data = shuffle(data)
data = data.reset_index(drop=True)
X = np.array(data.drop(data.iloc[:, [-1]], axis=1))  # Discard the class
y = np.array(data.iloc[:, [-1]])
feature_names = data.columns[:-1]
print("One-second eeg segments: {}".format(X.shape[0]))
print("Extracted features: {}".format(X.shape[1]))

# Feature Selection
est = LinearSVC(penalty="l1", dual=False)
c_grid = {'C': np.logspace(-3, 1, num=30)}
clf = GridSearchCV(estimator=est, param_grid=c_grid).fit(X,y)
reg_param = clf.best_params_['C']

lsvc = LinearSVC(C=reg_param, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_reduced = model.transform(X)
print("Selected features: {}".format(X_reduced.shape[1]))
print("Selected features names")
print(list(feature_names[model.get_support()]))
