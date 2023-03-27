import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.ensemble import ExtraTreesClassifier
import constants


def plot_features_for_all_classes(columns, importances):
    not_sorted = [(columns[i], importances[i]) for i in range(len(columns))]
    sorted_values = sorted(not_sorted, key=lambda x: x[1], reverse=True)
    x_values = [value[0] for value in sorted_values]
    y_values = [value[1] for value in sorted_values]
    plt.figure()
    plt.rcParams['figure.figsize'] = [16, 6]
    plt.title('Features\' importances')
    plt.bar(x_values, y_values, align="center")
    plt.xticks(rotation=30)
    # plt.xticks(range(len(result[i])), importances, rotation=90)
    # plt.xlim([-1, len(result[i])])
    plt.show()


def plot_features_by_class(result, ff):
    titles = ["Class 0", "Class 1"]
    for t, i in zip(titles, range(len(result))):
        plt.figure()
        plt.rcParams['figure.figsize'] = [16, 6]
        plt.title(t)
        plt.bar(range(len(result[i])), result[i].values(), align="center")
        plt.xticks(range(len(result[i])), ff[list(result[i].keys())], rotation=90)
        plt.xlim([-1, len(result[i])])
        plt.xticks(rotation=30)
        plt.show()


def class_feature_importance(X, Y, feature_importances):
    N, M = X.shape
    X = scale(X)

    out = {}
    for c in set(Y):
        out[c] = dict(
            zip(range(M), np.mean(X[Y == c, :], axis=0) * feature_importances)
        )

    return out


    ##get_ipython().run_line_magic('matplotlib', 'inline')
def importance_by_class(data):
    columns = list(data.columns)
    X = data.drop(columns=['Class'])
    y = data['Class']

    forest = ExtraTreesClassifier(random_state=1)
    forest.fit(X, y)

    importances = forest.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in forest.estimators_],
    #              axis=0)
    indices = np.argsort(importances)[::-1]
    feature_list = [X.columns[indices[f]] for f in range(X.shape[1])]  # names of features.

    ff = np.array(feature_list)
    features_dictionary = class_feature_importance(X, y, importances)

    plot_features_for_all_classes(X.columns, importances)
    plot_features_by_class(features_dictionary, X.columns)

    features_df = dict()
    for class_value in range(constants.number_of_classes):
        current_df = pd.DataFrame(columns=columns)
        for index in features_dictionary[class_value]:
            current_df[columns[index]] = features_dictionary[class_value][index]
        features_df[class_value] = current_df

    return features_df, ff
