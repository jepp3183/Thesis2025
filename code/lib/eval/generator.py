from enum import Enum
from lib.eval.eval_metrics import *

class metrics(Enum):
    Similarity = 0,
    Minimality = 1,
    Plausibility = 2,
    Validity = 3,
    Diversity = 4
    
def run(methods, centers, X, y, m = [metrics.Similarity, metrics.Minimality, metrics.Plausibility, metrics.Validity, metrics.Diversity]):
    results = {}
    for method in methods:
        results[method["name"]] = []

    for method in methods:
        cfs_data = method["counterfactuals"]
        
        for cf_data in cfs_data:
            metric = []
            cf = np.array(cf_data.cf)
            instance = X[cf_data.instance]
            target = cf_data.target
            instance_cluster = cf_data.instance_label

            if len(cf) == 0:
                results[method["name"]].append([[] for _ in range(len(m))])
                continue

            if metrics.Similarity in m:
                metric.append(cf_similarity(cf, instance))
            if metrics.Minimality in m:
                metric.append(cf_minimality(cf, instance))
            if metrics.Plausibility in m:
                metric.append(cf_plausibility(cf, target, X, y))
            if metrics.Validity in m:
                metric.append(cf_validity(cf, target, centers))
            if metrics.Diversity in m:
                div = cf_diversity(cf)
                metric.append(div)
            results[method["name"]].append(metric)
    return results, ["Similarity", "Minimality", "Plausibility", "Validity", "Diversity"]
