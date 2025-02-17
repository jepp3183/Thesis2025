import numpy as np
import pandas as pd

import lib.ext.baycon.baycon.bayesian_generator as baycon
import lib.ext.baycon.baycon.bayesian_generator as baycon
import lib.ext.baycon.baycon.time_measurement as time_measurement
from lib.ext.baycon.common.DataAnalyzer import *
from lib.ext.baycon.common.Target import Target
import seaborn as sns

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def execute(df, model, target, initial_instance_index, categorical_features=[], actionable_features=[]):
    y = df[[target.target_feature()]].values.ravel()
    X = df.drop([target.target_feature()], axis=1).values
    feature_names = df.columns[df.columns != target.target_feature()]

    run = 0
    data_analyzer = DataAnalyzer(X, y, feature_names, target, categorical_features, actionable_features)
    X, y = data_analyzer.data()
    initial_instance = X[initial_instance_index]
    initial_prediction = y[initial_instance_index]
    print("--- Executing... Initial Instance: {} Target: {} Run: {} ---".format(
        initial_instance_index,
        target.target_value_as_string(),
        run
    ))
    counterfactuals, ranker = baycon.run(initial_instance, initial_prediction, target, data_analyzer, model)
    predictions = np.array([])
    try:
        predictions = model.predict(counterfactuals)
    except ValueError:
        pass
    return counterfactuals, predictions, initial_instance, initial_prediction #, data_analyzer, ranker, model

def baycon_explainer(X, classifier, initial_point, target_label):
    y = classifier.predict(X)
    df = pd.DataFrame(np.column_stack((X, y)), columns=[f'x{i}' for i in range(X.shape[1])] + ['label'], dtype=float)

    t = Target(target_type="classification", target_feature="label", target_value=target_label)
    with HiddenPrints():
        cf, _predictions, _initial_instance, _initial_prediction = execute(df, classifier, t, initial_point)
    return cf

def baseline_explainer(X, cluster_labels, classifier, cf_method, initial_point, target_cluster):
    
    init_cluster = cluster_labels[initial_point]
    
    assert init_cluster != target_cluster, "Initial point is already in target cluster"

    # X = X[(cluster_labels == init_cluster) | (cluster_labels == target_cluster)]
    # cluster_labels = cluster_labels[(cluster_labels == init_cluster) | (cluster_labels == target_cluster)]

    classifier.fit(X, cluster_labels)
    print(f"Done training classifier. Score: {classifier.score(X, cluster_labels)}")

    initial_pred = classifier.predict(X[initial_point].reshape(1, -1))[0]
    assert initial_pred != target_cluster, "initial point classified in target cluster"

    cf = cf_method(X, classifier, initial_point, target_cluster)
    print(f"Found {len(cf)} counterfactuals")

    return cf