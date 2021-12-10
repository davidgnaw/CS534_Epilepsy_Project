# Created by: Daniela Chanci Arrubla
# Description: Application of SVM to the features extracted from each patient of the
# publicly available dataset "American Epilepsy Society Seizure Prediction Challenge" downloaded
# from kaggle

import numpy as np
import pandas as pd
import math
import os
from pathlib import Path
from matplotlib import pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
from sklearn import svm, metrics
from sklearn.metrics import RocCurveDisplay, auc, plot_confusion_matrix, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import warnings


def import_features(file_name, print_list):

    # Load the extracted features
    data = pd.read_csv(file_name)
    data = shuffle(data)
    data = data.reset_index(drop=True)
    X = data.drop(data.iloc[:, [-1]], axis=1) # Discard the class
    X = StandardScaler().fit_transform(X) # Data Standardization
    y = data.iloc[:, [-1]]
    feature_names = data.columns[:-1]

    # Print
    print_list.append("------------------------------------------------------------------------")
    print_list.append(subject)
    print_list.append("1. 30-seconds eeg segments: {}".format(X.shape[0]))
    print_list.append("2. Extracted features: {}".format(X.shape[1]))

    return X, y, feature_names, print_list


def balance_classes(X, y, print_list, method):

    # To oversample
    if method == "oversample":
        ros = RandomOverSampler(random_state=42)
        X_bal, y_bal = ros.fit_resample(X, np.array(y))

    # To undersample
    elif method == "undersample":
        # In case class 0 is greater than class 1
        if y.value_counts()[0] > y.value_counts()[1]:
            X_bal = np.zeros((y.value_counts()[1]*2, X.shape[1]))
            y_bal = np.zeros((y.value_counts()[1]*2, 1))
            index = 0
            counter = 0
            for i in range(len(np.array(y))):
                if np.array(y)[i] == 0:
                    if counter < y.value_counts()[1]:
                        X_bal[index,:] = X[i,:]
                        y_bal[index] = np.array(y)[i]
                        index += 1
                        counter += 1
                else:
                    X_bal[index, :] = X[i, :]
                    y_bal[index] = np.array(y)[i]
                    index += 1

        # In case class 1 is greater than class 0
        else:
            X_bal = np.zeros((y.value_counts()[0]*2, X.shape[1]))
            y_bal = np.zeros((y.value_counts()[0]*2, 1))
            index = 0
            counter = 0
            for i in range(len(np.array(y))):
                if np.array(y)[i] == 1:
                    if counter < y.value_counts()[0]:
                        X_bal[index, :] = X[i, :]
                        y_bal[index] = np.array(y)[i]
                        index += 1
                        counter += 1
                else:
                    X_bal[index, :] = X[i, :]
                    y_bal[index] = np.array(y)[i]
                    index += 1

        # Shuffle again
        data_bal = np.concatenate((X_bal, y_bal), axis=1)
        df_data_bal = pd.DataFrame(data_bal)
        data_bal = shuffle(df_data_bal)
        data_bal = data_bal.reset_index(drop=True)
        X_bal = data_bal.drop(data_bal.iloc[:, [-1]], axis=1)  # Discard the class
        X_bal = StandardScaler().fit_transform(X_bal)  # Data Standardization
        y_bal = np.array(data_bal.iloc[:, [-1]])

    else:
        print("Error")

    # Print
    print_list.append("3. Classes distribution before balancing:")
    print_list.append("  Counter(0: {}, 1: {})".format(y.value_counts()[0], y.value_counts()[1]))
    print_list.append("4. Classes distribution after balancing:")
    print_list.append("  Counter(0: {}, 1: {})".format(np.sum(y_bal == 0), np.sum(y_bal == 1)))

    return X_bal, y_bal, print_list


def split_dataset(X, y):

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def cross_validation(X_train, y_train, print_list, subject):

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=5)
    classifier = svm.SVC(kernel="linear", probability=True, random_state=42)

    tprs = []
    aucs = []
    fold_metrics = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    for i, (train, test) in enumerate(cv.split(X_train, y_train)):

        # Select best regularization parameter
        est = LinearSVC(penalty="l1", dual=False)
        c_grid = {'C': np.logspace(-3, 1, num=30)}
        clf = GridSearchCV(estimator=est, param_grid=c_grid).fit(X_train[train], y_train[train])
        reg_param = clf.best_params_['C']

        # Feature selection
        lsvc = LinearSVC(C=reg_param, penalty="l1", dual=False).fit(X_train[train], y_train[train])
        model = SelectFromModel(lsvc, prefit=True)
        X_reduced = model.transform(X_train[train])

        # Train classifier
        classifier.fit(X_reduced, y_train[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X_train[test][:,model.get_support()],
            y_train[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # Evaluation metrics per fold
        y_hat_fold = classifier.predict(X_train[test][:,model.get_support()])

        acc = metrics.accuracy_score(y_train[test], y_hat_fold)
        f1 = metrics.f1_score(y_train[test], y_hat_fold, average='macro')
        precision = metrics.precision_score(y_train[test], y_hat_fold, average='macro')
        recall = metrics.recall_score(y_train[test], y_hat_fold, average='macro')
        fold_metrics.append([i, acc, precision, recall, f1])


    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="ROC Curve",
    )
    ax.legend(loc="lower right")

    # Save AUROC
    fig_dir = os.path.join("F:/Users/user/Desktop/EMORY/Classes/Fall_2021/CS_534/Project/Prediction/Figures_SVM", "AUROC_"+subject+".png")
    plt.savefig(fig_dir)
    # plt.show()

    print_list.append("5. Evaluation metrics CV:")
    print_list.append("  {}".format(["Fold", "Accuracy", "Precision", "Recall", "F1-Score"]))
    for elem in fold_metrics:
        print_list.append("  {}".format(elem))

    return print_list


def evaluate_model(X_train, X_test, y_train, y_test, print_list, subject):

    # Select best regularization parameter
    est = LinearSVC(penalty="l1", dual=False)
    c_grid = {'C': np.logspace(-3, 1, num=30)}
    clf = GridSearchCV(estimator=est, param_grid=c_grid).fit(X_train, y_train)
    reg_param = clf.best_params_['C']

    # Feature selection
    lsvc = LinearSVC(C=reg_param, penalty="l1", dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    X_reduced = model.transform(X_train)

    # Print
    print_list.append("6. Selected features: {}".format(X_reduced.shape[1]))
    print_list.append("7. Selected features names:")
    for idx in range(math.ceil(X_reduced.shape[1] / 3)):
        print_list.append("  {}".format(list(feature_names[model.get_support()])[idx * 3:idx * 3 + 3]))

    # Train SVM
    svm_model = svm.SVC(kernel="linear", probability=True, random_state=42)
    svm_model.fit(X_reduced, y_train)

    # Obtain predictions
    y_hat = svm_model.predict(X_test[:,model.get_support()])

    # Metrics
    acc = metrics.accuracy_score(y_test, y_hat)
    f1 = metrics.f1_score(y_test, y_hat, average='macro')
    precision = metrics.precision_score(y_test, y_hat, average='macro')
    recall = metrics.recall_score(y_test, y_hat, average='macro')
    auroc = metrics.roc_auc_score(y_test, svm_model.decision_function(X_test[:,model.get_support()]), average='macro')
    RMSE = metrics.mean_squared_error(y_test, y_hat, squared=False)

    # Print
    print_list.append("8. Evaluation Metrics:")
    print_list.append("  Accuracy: {}".format(acc))
    print_list.append("  Precision: {}".format(precision))
    print_list.append("  Recall: {}".format(recall))
    print_list.append("  F1-score: {}".format(f1))
    print_list.append("  AUROC: {}".format(auroc))
    print_list.append("  RMSE: {}".format(RMSE))

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_hat, normalize='true')

    # Print
    print_list.append("9. Confusion Matrix")
    print_list.append("  {},{}".format(conf_mat[0],conf_mat[1]))

    # Plot confusion Matrix
    plot_confusion_matrix(svm_model, X_test[:,model.get_support()], y_test, display_labels=("Interictal", "Ictal"),
                          cmap=plt.cm.Blues, normalize="true")

    # Save Confusion Matrix
    fig_dir = os.path.join("F:/Users/user/Desktop/EMORY/Classes/Fall_2021/CS_534/Project/Prediction/Figures_SVM", "CM_"+subject+".png")
    plt.savefig(fig_dir)
    # plt.show()

    print_list.append("\n")

    return print_list


if __name__ == '__main__':

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Initialize list to print
    toprint = []
    toprint.append("SVM Prediction Summary")
    toprint.append("\n")

    # Define parent directory
    parent_Dir = Path("F:/Users/user/Desktop/EMORY/Classes/Fall_2021/CS_534/Project/Prediction/Extracted_Features")
    files = list(parent_Dir.glob('*.csv'))

    # Loop through subjects
    for filename in files:
        # Define subject name
        subject = str(filename).split("\\")[-1].split(".")[0]

        # Import extracted features
        X, y, feature_names, toprint = import_features(filename, toprint)

        # Balance classes
        X_bal, y_bal, toprint = balance_classes(X, y, toprint, 'undersample')

        # Obtain training and testing set
        X_train, X_test, y_train, y_test = split_dataset(X_bal, y_bal)

        # Cross-validation 5-folds
        toprint = cross_validation(X_train, y_train, toprint, subject)

        # Evaluation metrics and confusion matrix
        toprint = evaluate_model(X_train, X_test, y_train, y_test, toprint, subject)


    # Write file
    textfile = open(r"F:\Users\user\Desktop\EMORY\Classes\Fall_2021\CS_534\Project\Prediction\Results_SVM.txt", "w")
    for elem in toprint:
        print(elem)
        textfile.write(str(elem) + "\n")
    textfile.close()

