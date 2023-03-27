import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def get_test_train_data(dataframe, class_index):
    x = dataframe.drop(columns=['Class'])
    y = dataframe['Class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)
    return x_train, x_test, y_train, y_test


def calculate_knn(x_train, x_test, y_train, y_test):
    # x_train, x_test, y_train, y_test = get_test_train_data(dataframe, class_index)
    # pd.DataFrame(x_train).to_excel("x_train2.xlsx")
    standard_scaler = StandardScaler()
    x_train = standard_scaler.fit_transform(x_train)
    x_test = standard_scaler.fit_transform(x_test)

    dataframe_size = len(y_test)
    k = int(round(math.sqrt(len(y_test))))
    k += 1 * (dataframe_size % 2 == math.sqrt(len(y_test)) % 2)

    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric="euclidean")
    classifier.fit(x_train, y_train)

    y_predicted = classifier.predict(x_test)

    cm = confusion_matrix(y_test, y_predicted)
    print(cm)

    print("f1 score: {0}".format(f1_score(y_test, y_predicted)))
    print("accuracy score: {0}".format(accuracy_score(y_test, y_predicted)))

    output_df = pd.DataFrame(x_test)
    output_df['Class'] = y_predicted
    return cm, f1_score(y_test, y_predicted),  accuracy_score(y_test, y_predicted), output_df

