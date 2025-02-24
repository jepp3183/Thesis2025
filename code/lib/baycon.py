import numpy as np
import pandas as pd

import lib.ext.baycon.baycon.bayesian_generator as baycon
import lib.ext.baycon.baycon.bayesian_generator as baycon
import lib.ext.baycon.baycon.time_measurement as time_measurement
from lib.ext.baycon.common.DataAnalyzer import *
from lib.ext.baycon.common.Target import Target
from lib.util import HiddenPrints
from lib.ext.baycon.common.ScoreCalculator import ScoreCalculator

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
    base_calculator = ScoreCalculator(initial_instance, initial_prediction, target, data_analyzer)
    counterfactuals, ranker = baycon.run(initial_instance, initial_prediction, target, data_analyzer, model, base_calculator)
    predictions = np.array([])
    try:
        predictions = model.predict(counterfactuals)
    except ValueError:
        pass
    return counterfactuals, predictions, initial_instance, initial_prediction #, data_analyzer, ranker, model

def baycon_explainer(X, classifier, initial_point, target_label, quiet=False):
    y = classifier.predict(X)
    df = pd.DataFrame(np.column_stack((X, y)), columns=[f'x{i}' for i in range(X.shape[1])] + ['label'], dtype=float)

    t = Target(target_type="classification", target_feature="label", target_value=target_label)

    if quiet:
        with HiddenPrints():
            cf, _predictions, _initial_instance, _initial_prediction = execute(df, classifier, t, initial_point)
    else:
        cf, _predictions, _initial_instance, _initial_prediction = execute(df, classifier, t, initial_point)
    return cf