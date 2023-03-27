import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pca import plot_3d

import constants

path = "creditcard.csv"


def get_prediction(x, y):
    cluster = AgglomerativeClustering(n_clusters=constants.number_of_classes, affinity='euclidean', linkage='ward')
    cluster.fit(x)
    print(accuracy_score(y, cluster.labels_))  # 0.9931680655865703
    y_hc = cluster.fit_predict(x)

    x_df = pd.DataFrame(x, columns=[0, 1, 2])
    x_df['Class'] = y_hc
    plot_3d(x_df)

    plt.scatter(x[y_hc == 0], x[y_hc == 0], x[y_hc == 0], c='cyan')
    plt.scatter(x[y_hc == 1], x[y_hc == 1], x[y_hc == 1], c='yellow')
    plt.show()


def plot_dendrogram(dataframe):
    np.set_printoptions(precision=4, suppress=True)
    plt.figure(figsize=(10, 3))
    plt.style.use('seaborn-whitegrid')

    # x = normalize(dataframe.iloc[:, :-1].values)
    x = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values

    # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.01)

    # linkg = linkage(x, "ward")
    # dendrogram(linkg, truncate_mode='lastp')
    # plt.xlabel("Sample")
    # plt.show()
    get_prediction(x, y)
