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


path = "creditcard.csv"

dataset = pd.read_csv(path)
print(dataset.head())

x = dataset.iloc[:, 0:30]
y = dataset.iloc[:, 30]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.fit_transform(x_test)

dataset_size = len(y_test)
k = int(round(math.sqrt(len(y_test))))
k += 1 * (dataset_size % 2 == math.sqrt(len(y_test)) % 2)

classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric="euclidean")
classifier.fit(x_train, y_train)

y_predicted = classifier.predict(x_test)
# print(y_predicted)

cm = confusion_matrix(y_test, y_predicted)
print(cm)

print("f1 score: {0}".format(f1_score(y_test, y_predicted)))
print("accuracy score: {0}".format(accuracy_score(y_test, y_predicted)))

