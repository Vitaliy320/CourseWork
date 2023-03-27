import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier, plot_importance
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, classification_report


def plot_3d(dataframe):
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    # xline = dataframe["principal component 1"]
    # yline = dataframe["principal component 2"]
    # zline = dataframe["principal component 3"]
    # ax.plot3D(xline, yline, zline, 'gray')

    colors = ['b', 'r']
    targets = [0, 1]
    # dataframe = dataframe.head(len(dataframe["principal component 1"]) // 10)
    for target, color in zip(targets, colors):
        indicesToKeep = dataframe['Class'] == target
        ax.scatter(dataframe.loc[indicesToKeep, 0],
                   dataframe.loc[indicesToKeep, 1],
                   dataframe.loc[indicesToKeep, 2], c=color, s=50)
    ax.legend(targets)
    ax.grid()

    # Data for three-dimensional scattered points
    xdata = dataframe[0]
    ydata = dataframe[1]
    zdata = dataframe[2]
    ax.scatter3D(xdata[:1], ydata[:1], zdata[:1])
    plt.show()


def visualise_weights(df, n_dimensions, columns_names):
    for dim in range(n_dimensions):
        plt.figure()
        plt.rcParams['figure.figsize'] = [16, 6]
        plt.title('Features\' influence on the primary component {0}'.format(dim))
        plt.bar(list(df.columns), df.iloc[dim, :].values.flatten().tolist(), align="center")
        plt.xticks(rotation=30)
        # plt.xlim([-1, len(result[i])])
        plt.show()


def get_significant_features_by_component(weights_t_classes):
    feature_indices_by_class = dict()
    for index, weights_t in enumerate(weights_t_classes):
        features_indices_by_component = dict()
        components = list(weights_t.columns)
        for component_index, component in enumerate(components):
            current_component = weights_t[component]
            max_value = max(current_component)

            for feature_index, feature_value in enumerate(current_component):
                if feature_value >= 0.75 * max_value or \
                        (feature_value < 0 and abs(feature_value) >= 0.75 * max_value):
                    if component_index in features_indices_by_component:
                        features_indices_by_component[component_index].append(feature_index)
                    else:
                        features_indices_by_component[component_index] = [feature_index]
        feature_indices_by_class[index] = features_indices_by_component
    return feature_indices_by_class


def get_component_weight_by_class(features_indices_by_component, weights_t_classes):
    component_weights_by_class = dict()
    for class_value in (0, 1):
        current_class = dict()
        for key in list(features_indices_by_component[class_value].keys()):
            component = list(weights_t_classes[class_value][str(key)])
            sum_all = sum([value**2 for value in component])
            sum_significant = sum([value**2 for index, value in enumerate(component)
                                   if index in features_indices_by_component[class_value][key]])
            current_class[key] = sum_significant / sum_all
        component_weights_by_class[class_value] = current_class
    return component_weights_by_class


def get_components_weights(dataframe):
    pca = PCA(n_components=3)
    x_all = dataframe.drop(columns=["Class"])
    x_all = StandardScaler().fit_transform(x_all)
    principal_components = pd.DataFrame(pca.fit_transform(x_all))
    principal_components['Class'] = dataframe['Class']
    weights_t_classes = []
    for class_value in [0, 1]:
        current_dataframe = dataframe[dataframe['Class'] == class_value]
        x = current_dataframe.drop(columns=["Class"])
        y = current_dataframe["Class"]
        x_columns = list(x.columns)
        x = StandardScaler().fit_transform(x)

        pca = PCA(n_components=3)
        _ = pca.fit_transform(x)
        weights = pd.DataFrame()
        weights_t = pd.DataFrame()
        for i in range(len(x_columns)):
            feature_weights = [float(pca.components_[j][i]) for j in range(3)]
            weights[x_columns[i]] = feature_weights
        for i in range(3):
            feature_weights = [float(pca.components_[i][j]) for j in range(len(x_columns))]
            weights_t[str(i)] = feature_weights
        weights_t_classes.append(weights_t)
        # principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2',
        #                                                                 'principal component 3'])
        #
        # final_df = pd.concat([principal_df, y], axis=1)
    return principal_components, weights, weights_t_classes


def plot_2d(final_df):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0, 1]
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = final_df['Class'] == target
        ax.scatter(final_df.loc[indicesToKeep, 'principal component 1'],
                   final_df.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()

    plt.show()


def visualise(x_train, x_test, y_train, y_test):
    params = {
        'max_depth': [6, 8, 10, 14],
        'learning_rate': [0.1, 0.15, 0.2],
        'num_leaves': [10, 20, 30]
    }

    lgbm = LGBMClassifier(n_estimators=100, boost_from_average=False)
    lgbm.fit(x_train, y_train,)
    # best = lgbm_gs.best_estimator_
    # lgbm_gs_pred = best.predict(x_test)
    # lgbm_gs_pred_proba = best.predict_proba(x_test)[:, 1]
    # print(f'정확도:{accuracy_score(y_test,lgbm_gs_pred)*100}\nAUC:{roc_auc_score(y_test,lgbm_gs_pred_proba)}\n{classification_report(y_test,lgbm_gs_pred)}')
    plt.rcParams["figure.figsize"] = (10, 8)
    plot_importance(lgbm, color='Navy')
    plt.show()
    v = 1
