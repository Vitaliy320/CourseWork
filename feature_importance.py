# logistic regression for feature importance
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler, scale
import numpy as np
import matplotlib.pyplot as plt
import constants
from collections import OrderedDict


def plot_features_by_class(features_df, columns_names):
    result = dict()
    # result = {index: list(value.values()) for index, value in enumerate(features_df)}#
    for df_index, df in enumerate(features_df):
        result[df_index] = df.values
    titles = ["Class 0", "Class 1"]
    for i, title in enumerate(titles):
        plt.figure()
        plt.rcParams['figure.figsize'] = [16, 6]
        plt.title(title)
        plt.bar(columns_names, features_df[i].values.flatten().tolist(), color="r", align="center")
        # plt.xlim([-1, len(result[i])])
        plt.show()


def normalise_values(values_list):
    list_sum = np.sum(values_list)
    return [value / list_sum for value in values_list]


def get_importance_by_class(x, y, importances):
    features_by_class = dict()
    x = scale(x)
    for class_value in range(constants.number_of_classes):
        features_by_class[class_value] = np.mean(x[y == class_value], axis=0) * importances
    return features_by_class


def regression_features(dataframe):
    x = dataframe.drop(columns=['Class'])
    y = dataframe['Class']
    columns_names = list(x.columns)

    model = LogisticRegression()
    model.max_iter = len(y)

    model.fit(x, y)

    y_pred = model.predict(x)
    print(model.score(x, y))

    importances = model.coef_[0]
    features_by_class = get_importance_by_class(x, y, importances)
    features_dataframes = []
    for key in features_by_class:
        current_dataframe = pd.DataFrame([features_by_class[key]], columns=columns_names)
        features_dataframes.append(current_dataframe)
    plot_features_by_class(features_dataframes, columns_names)
    # for i, v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    features = []
    for index, column in enumerate(x.columns):
        features.append((column, importances[index]))

    features_sorted = sorted(features, key=lambda tup: tup[1], reverse=True)
    pyplot.bar([col[0] for col in features_sorted], [col[1] for col in features_sorted], color="r")
    pyplot.show()
