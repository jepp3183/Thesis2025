import numpy as np
from lib.eval.tools import euclid_dis
from lib.eval.tools import center_prediction
from sklearn.neighbors import LocalOutlierFactor

def cf_similarity(cf, instance, dis = euclid_dis) -> float:
    assert len(cf.shape) == 2
    return np.array([dis(instance, cf_i) for cf_i in cf])

def cf_validity(cf, instance_cluster, centers, eval = center_prediction) -> float:
    assert len(cf.shape) == 2

    return float(instance_cluster) != eval(cf, centers)

def cf_plausibility(cf, target, X, y) -> float:
    assert len(cf.shape) == 2
    
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    target_points = X[y==target]
    
    clf.fit(target_points)
    pred = clf.score_samples(cf)
    return pred

def cf_minimality(cf, instance) -> float:
    assert len(cf.shape) == 2

    res = np.equal(cf, instance).mean(axis=1)
    if len(res) == 1:
        return res[0]
    return res