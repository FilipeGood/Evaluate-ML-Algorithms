from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor, Lasso, Ridge, ElasticNet
import pandas as pd
from sklearn.model_selection import cross_validate
import argparse
import matplotlib.pyplot as plt
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', type=str, help="Path to dataset")
ap.add_argument('-p', '--problem_type', type=str, help="Problem type - classification or regression")
ap.add_argument('-t', '--target', type=str, help="Target column")
args = vars(ap.parse_args())


def create_baseline_models(problem_type):
    models = list()
    if problem_type == 1:
        # Classification Problems
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('LR', LogisticRegression()))
        models.append(('DT', DecisionTreeClassifier()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVC', SVC()))
        models.append(( 'ADA',AdaBoostClassifier()))
        models.append(( 'GB',GradientBoostingClassifier()))
        models.append(('RF' ,RandomForestClassifier()))
        models.append(('ET',ExtraTreesClassifier()))
        models.append(('SGD',SGDClassifier()))
        models.append(('MultiNB',MultinomialNB()))
        return models

    else:
        # Regression Problems
        models.append(( 'LR', LinearRegression()))
        models.append(('SGD' ,SGDRegressor()))
        models.append(('ET' ,ExtraTreesRegressor()))
        models.append(( 'GB',GradientBoostingRegressor()))
        models.append(( 'RF',RandomForestRegressor()))
        models.append(('SVR' ,SVR()))
        models.append(( 'Lasso',Lasso()))
        models.append(( 'Ridge',Ridge()))
        models.append(( 'ElasticNet',ElasticNet()))
        models.append(( 'KNN',KNeighborsRegressor()))
        models.append(('ADA' ,AdaBoostRegressor()))

        models.append(('DT' ,DecisionTreeRegressor()))
    return models


def evaluate_models(X_train, y_train, cv =5, models = None, problem_type = 1, metrics = None):
    if models is None:
        models = create_baseline_models(problem_type)

    if metrics is None:
        classifcation_metris =  ['roc_auc', 'f1','accuracy']
        regression_metrics = ['neg_mean_squared_log_error','neg_mean_squared_error', 'r2' ]
        metrics = classifcation_metris if problem_type==1 else regression_metrics

    results = pd.DataFrame()

    for name, model in models:
        score = pd.DataFrame(cross_validate(model, X_train, y_train, cv=cv, scoring=metrics))
        mean = score.mean().rename('{}_mean'.format)
        std = score.std().rename('{}_std'.format)
        results[name] = pd.concat([mean, std], axis = 0)

    return results

def plot_results(results, metric = 'accuracy'):

    results = results.transpose()
    plt.figure(1)
    results['fit_time_mean'].plot(kind='bar', title='Fit time mean', grid=True, color=plt.cm.Paired(np.arange(len(results))))
    plt.figure(2)
    results[f'test_{metric}_mean'].plot(kind='bar', title=f'Test {metric} mean', grid=True, color=plt.cm.Paired(np.arange(len(results))))

    plt.show()


if __name__ == "__main__":
    dataset_path  = args['dataset']
    problem_type = args['problem_type']
    target = args['target']
    problem_type = 1 if problem_type == 'classification' else 0

    df = pd.read_csv(dataset_path)
    X = df.drop(target, 1)
    y = df[target].values

    models = create_baseline_models(problem_type)
    results = evaluate_models(X, y, models=models)

    print('\t\t Results: \n', results)

    plot_results(results)