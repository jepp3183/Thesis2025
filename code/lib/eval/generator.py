from enum import Enum
from lib.eval.eval_metrics import *

class metrics(Enum):
    Similarity = 0,
    Minimality = 1,
    Plausibility = 2,
    Validity = 3
    
def run(methods, centers, X, y, m = [metrics.Similarity, metrics.Minimality, metrics.Plausibility, metrics.Validity]):
    results = {}
    for method in methods:
        results[method["name"]] = []

    for method in methods:
        cfs_data = method["counterfactuals"]
        
        for cf_data in cfs_data:
            metric = []
            cf = cf_data.cf
            instance = X[cf_data.instance]
            target = cf_data.target
            instance_cluster = cf_data.instance_label

            if metrics.Similarity in m:
                metric.append(cf_similarity(cf, instance))
            if metrics.Minimality in m:
                metric.append(cf_minimality(cf, instance))
            if metrics.Plausibility in m:
                metric.append(cf_plausibility(cf, target, X, y))
            if metrics.Validity in m:
                metric.append(cf_validity(cf, instance_cluster, centers))
            results[method["name"]].append(metric)
    return results
