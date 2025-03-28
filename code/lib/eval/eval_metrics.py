import numpy as np
from lib.eval.tools import euclid_dis
from lib.eval.tools import center_prediction
from sklearn.neighbors import LocalOutlierFactor
import random

def cf_similarity(cf, instance, dis = euclid_dis) -> float:
    assert len(cf.shape) == 2
    return np.array([dis(instance, cf_i) for cf_i in cf])

def cf_validity(cf, target_cluster, centers, eval = center_prediction) -> float:
    assert len(cf.shape) == 2

    dists = np.linalg.norm(cf[:, None] - centers, axis=2)
    print(f"dists shape: {dists.shape}")
    pred = np.argmin(dists, axis = 1) 
    print(f"pred shape: {pred.shape}")
    print(f"pred: {pred}")

    r = pred == int(target_cluster)
    print(f"r shape: {r.shape}")
    return r

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

def cf_diversity(cf, per=0.05, dis = euclid_dis):
    assert len(cf.shape) == 2

    m = np.empty((len(cf), len(cf)))
    for i in range(len(cf)):
        m[:,i] = [(1.0 / (1 + dis(cf[i], cf_t))) for cf_t in cf]
        m[i,i] += random.uniform(per,per*2)

    return np.linalg.det(m)