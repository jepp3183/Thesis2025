from enum import Enum
from lib.eval.eval_metrics import *
from tqdm import tqdm

class metrics(Enum):
    Similarity = 0,
    Minimality = 1,
    Plausibility = 2,
    Validity = 3,
    Diversity = 4,
    Invalidation = 5,
    Runtime = 6
    
def run(
        method, 
        centers, 
        X, 
        y, 
        runtimes,
        m = [metrics.Similarity, metrics.Minimality, metrics.Plausibility, metrics.Validity, metrics.Diversity, metrics.Invalidation, metrics.Runtime],
        remove_invalid = True,
    ):
    results = []
    print("Starting on: " + method["name"])
    if remove_invalid:
        print("Removing invalid counterfactuals!!!")
        
        
    cfs_data = method["counterfactuals"]

    for i, cf_data in enumerate(cfs_data):
        metric = []
        cf_original = np.array(cf_data.cf)
        instance = X[cf_data.instance]
        target = cf_data.target

        if remove_invalid:
            dists = np.linalg.norm(cf_original[:, None] - centers, axis=2)
            pred = np.argmin(dists, axis = 1) 
            r = pred == int(target)
            cf = cf_original[r]
        else:
            cf = cf_original


        if len(cf) == 0:
            results.append([
                [],
                [],
                [],
                cf_validity(cf_original, target, centers),
                [],
                [],
                runtimes[i]
            ])
            continue

        if metrics.Similarity in m:
            metric.append(cf_similarity(cf, instance))
        if metrics.Minimality in m:
            metric.append(cf_minimality(cf, instance))
        if metrics.Plausibility in m:
            metric.append(cf_plausibility(cf, target, X, y))
        if metrics.Validity in m:
            metric.append(cf_validity(cf_original, target, centers))
        if metrics.Diversity in m:
            metric.append(cf_diversity(cf))
        if metrics.Invalidation in m:
            metric.append(cf_counterfactual_invalidation(cf_original, X, cf_data.instance, centers, target))
        if metrics.Runtime in m:
            metric.append(runtimes[i])
        

        results.append(metric)
    return results, returnNames()

def returnNames():
    return ["Similarity", "Minimality", "Plausibility", "Validity", "Diversity", "Invalidation", "Runtime"]