import numpy as np
from lib.eval.tools import euclid_dis
from lib.eval.tools import center_prediction
from lib.eval.tools import minimality_metric
from sklearn.neighbors import LocalOutlierFactor

euclid_par = lambda a,b : euclid_dis(a,b)
prediction_par = lambda a, centers : center_prediction(a, centers)
mini_par = lambda a,b : minimality_metric(a,b)

def cf_similarity(instance, cf, dis = euclid_par) -> float:
    return dis(instance, cf)

def cf_validity(cf, instance_cluster, centers, eval = prediction_par) -> float:
    return float(instance_cluster) != eval(cf, centers)

def cf_plausibility(cf, target, X, y) -> float:
    assert len(cf.shape) == 2
    
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    target_points = X[y==target]
    
    clf.fit(target_points)
    pred = clf.score_samples(cf)
    return pred

def cf_minimality(cf, instance, eval = mini_par) -> float:
    return eval(cf, instance)