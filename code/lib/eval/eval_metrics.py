import numpy as np
from lib.eval.tools import euclid_dis
from lib.eval.tools import center_prediction
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
import random

def cf_similarity(cf, instance, dis = euclid_dis) -> float:
    assert len(cf.shape) == 2
    return np.array([dis(instance, cf_i) for cf_i in cf])

def cf_validity(cf, target_cluster, centers, eval = center_prediction) -> float:
    assert len(cf.shape) == 2

    dists = np.linalg.norm(cf[:, None] - centers, axis=2)
    # print(f"dists shape: {dists.shape}")
    pred = np.argmin(dists, axis = 1) 
    # print(f"pred shape: {pred.shape}")
    # print(f"pred: {pred}")

    r = pred == int(target_cluster)
    # print(f"r shape: {r.shape}")
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
    return res

def cf_diversity(cf, per=0.05, dis = euclid_dis):
    assert len(cf.shape) == 2

    m = np.empty((len(cf), len(cf)))
    for i in range(len(cf)):
        m[:,i] = [(1.0 / (1 + dis(cf[i], cf_t))) for cf_t in cf]
        m[i,i] += random.uniform(per,per*2)

    return np.linalg.det(m)

def cf_counterfactual_invalidation(cf, X, instance, centers, target, random_state=None, return_kmeans=False):
    assert len(cf.shape) == 2

    kmeans_list = []

    result = np.zeros(len(cf))
    for i in range(len(cf)):
        cf_temp = cf[i]

        new_kmeans = KMeans(n_clusters=len(centers), init=centers, random_state=random_state, n_init=1)
        new_X = X.copy()
        new_X[instance] = cf_temp
        new_kmeans.fit(new_X)
        new_centers = new_kmeans.cluster_centers_

        if return_kmeans:
            kmeans_list.append(new_kmeans)   
        formated_cf = np.array([cf_temp])
        result[i] = cf_validity(formated_cf, target, new_centers)
    if return_kmeans:
        return result, kmeans_list
    return result