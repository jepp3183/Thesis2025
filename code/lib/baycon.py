import numpy as np
import pandas as pd

from lib.score_calculator_kmeans import ScoreCalculatorKmeans
from lib.score_calculator_model_agnostic import ScoreCalculatorModelAgnostic
import lib.ext.baycon.baycon.bayesian_generator as baycon
import lib.ext.baycon.baycon.bayesian_generator as baycon
import lib.ext.baycon.baycon.time_measurement as time_measurement
from lib.ext.baycon.common.DataAnalyzer import *
from lib.ext.baycon.common.Target import Target
from lib.util import HiddenPrints
from lib.ext.baycon.common.ScoreCalculator import ScoreCalculator

def execute(df, model, target, initial_instance_index, categorical_features=[], actionable_features=[], amount_of_coreset_points=100):
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

    target_cluster_center = None
    min_target_cluster_distance = None
    max_target_cluster_distance = None

    # Initialize ScoreCalculator Classification / Clustering
    initial_instance_f = initial_instance.astype(float)   # np operations need same type object to compute!
    if target.target_type() == Target.TYPE_CLASSIFICATION or target.target_type() == Target.TYPE_REGRESSION:
        score_calculator = ScoreCalculator(initial_instance, initial_prediction, target, data_analyzer)
    elif target.target_type() == Target.TYPE_CLUSTERING_KMEANS:
        # point_pred = int(model.predict([X[initial_instance_index]])[0])
        target_cluster_center = model.cluster_centers_[target._target_value]
        
        target_cluster_indices = np.where(y == target._target_value)

        min_target_cluster_distance = np.min([np.linalg.norm(i-target_cluster_center) for i in X[target_cluster_indices]])
        max_target_cluster_distance = np.max([np.linalg.norm(i-target_cluster_center) for i in X[target_cluster_indices]])
        # max_target_cluster_distance = np.max([np.linalg.norm(i-target_cluster_center) for i in X])

        print(target_cluster_center)
        print(min_target_cluster_distance)
        print(max_target_cluster_distance)

        base_calculator = ScoreCalculator(initial_instance, initial_prediction, target, data_analyzer)
        score_calculator = ScoreCalculatorKmeans(initial_instance, initial_prediction, target, data_analyzer, base_calculator, min_target_cluster_distance, max_target_cluster_distance, target_cluster_center)
    elif target.target_type() == Target.TYPE_MODEL_AGNOSTIC:
        # amount_of_coreset_points = 100

        base_calculator = ScoreCalculator(initial_instance, initial_prediction, target, data_analyzer)
        score_calculator = ScoreCalculatorModelAgnostic(initial_instance, initial_prediction, target, data_analyzer, base_calculator, amount_of_coreset_points, X, y)

    #base_calculator = ScoreCalculator(initial_instance, initial_prediction, target, data_analyzer)
    counterfactuals, ranker = baycon.run(initial_instance_f, initial_prediction, target, data_analyzer, model, score_calculator)
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