import numpy as np
# logistic regression for feature importance
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn import tree
from subprocess import check_call
import matplotlib.pyplot as plt
from subprocess import call
from IPython.display import SVG
from graphviz import Source
from sklearn.metrics import mean_squared_error
from sklearn.tree import _tree
import graphviz
import constants


def get_rules(tree, feature_names, class_names, X, y):
    # Fit the classifier with max_depth=3
    clf = DecisionTreeClassifier(max_depth=3, random_state=1234)
    model = clf.fit(X, y)

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


def write_to_file(file_name, rules):
    with open(file_name, 'w') as f:
        for index, line in enumerate(rules):
            f.write(str(index + 1) + '. ' + line)
            f.write('\n')


def create_decision_tree(x_train, x_test, y_train, y_test):
    """
    !dot -Tpng tree.dot -o tree.png
    """
    # x_train = pd.DataFrame()
    # for column in x_train_high_precision.columns:
    #     x_train[column] = x_train_high_precision[column].round(decimals=2)
    dt_model = tree.DecisionTreeRegressor(max_depth=8)
    dt_model.fit(x_train, y_train)
    print(dt_model.score(x_train, y_train))
    y_pred = dt_model.predict(x_test)

    rules = get_rules(dt_model, x_train.columns, [str(i) for i in range(constants.number_of_classes)], x_train, y_train)
    write_to_file('rules.txt', rules)
    v = 1
    # print(np.sqrt(mean_squared_error(y_test, y_pred)))
    # plt.plot(y_pred)
    # plt.show()
    # dt_model.score(x_test, y_test)
    # dt_model.predict(x_test)
    # y_pred = dt_model.predict_proba(x_test)
    # new_y = []
    # for i in range(len(y_pred)):
    #     if y_pred[i] < 0.6:
    #         new_y.append(0)
    #     else:
    #         new_y.append(1)
    # acc_score = accuracy_score(y_test, new_y)
    # decision_tree = tree.export_graphviz(dt_model, out_file='tree.dot',
    #                                      feature_names=x_train.columns, max_depth=2, filled=True)

    dotfile = open("tree2.dot", 'w')
    tree.export_graphviz(dt_model, out_file=dotfile, feature_names=x_train.columns)
    dotfile.close()
    graph = Source(tree.export_graphviz(dt_model, out_file=None, feature_names=x_train.columns))
    graph.format = 'png'
    graph.render('dtree_render', view=True)





