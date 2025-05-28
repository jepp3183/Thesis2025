import numpy as np
from lib.eval.tools import euclid_dis
from lib.eval.tools import center_prediction
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
import random

def cf_similarity(cf, instance, dis = euclid_dis):
    assert len(cf.shape) == 2
    return np.array([dis(instance, cf_i) for cf_i in cf])

def cf_validity(cf, target_cluster, centers, eval = center_prediction):
    assert len(cf.shape) == 2

    dists = np.linalg.norm(cf[:, None] - centers, axis=2)
    pred = np.argmin(dists, axis = 1) 
    return pred == int(target_cluster)

def cf_plausibility(cf, target, X, y):
    assert len(cf.shape) == 2
    
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    target_points = X[y==target]
    
    clf.fit(target_points)

    pred = clf.score_samples(cf)
    return pred

def cf_minimality(cf, instance):
    assert len(cf.shape) == 2

    res = np.equal(cf, instance).mean(axis=1)
    return res

def cf_diversity(cf, per=0.05, dis = euclid_dis):
    assert len(cf.shape) == 2

    if len(cf) == 0:
        return None

    m = np.empty((len(cf), len(cf)))
    for i in range(len(cf)):
        m[:,i] = [(1.0 / (1 + dis(cf[i], cf_t))) for cf_t in cf]
        m[i,i] += random.uniform(per,per*2)

    return np.linalg.det(m)

def cf_counterfactual_invalidation(cf, X, instance, centers, target, random_state=None, correction=False):
    assert len(cf.shape) == 2

    result = []
    for i,cf_temp  in enumerate(cf):

        pre_validity = cf_validity(np.array([cf_temp]), target, centers).mean()

        if pre_validity == correction:
            continue
        
        new_kmeans = KMeans(n_clusters=len(centers), init=centers, random_state=random_state)
        new_X = X.copy()
        new_X[instance] = cf_temp
        new_kmeans.fit(new_X)
        new_centers = new_kmeans.cluster_centers_

        formated_cf = np.array([cf_temp])
        post_valid = cf_validity(formated_cf, target, new_centers).mean()

        result.append(post_valid == correction)

    if len(result) == 0:
        return None
    return np.array(result)

def cf_percent_explained(cf, target, centers):
    assert len(cf.shape) == 2

    val = cf_validity(cf, target, centers).mean()
    return val > 0
