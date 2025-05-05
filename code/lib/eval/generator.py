from enum import Enum
from lib.eval.eval_metrics import *
from tqdm import tqdm

class metrics(Enum):
    Similarity = 0,
    Minimality = 1,
    Plausibility = 2,
    Validity = 3,
    Diversity = 4
    
def run(
        method, 
        centers, 
        X, 
        y, 
        m = [metrics.Similarity, metrics.Minimality, metrics.Plausibility, metrics.Validity, metrics.Diversity],
        remove_invalid = True,
    ):
    results = []
    print("Starting on: " + method["name"])
    if remove_invalid:
        print("Removing invalid counterfactuals!!!")
        
        
    cfs_data = method["counterfactuals"]

    for cf_data in tqdm(cfs_data):
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
            results.append([[] for _ in range(len(m))])
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
            div = cf_diversity(cf)
            metric.append(div)

        results.append(metric)
    return results, returnNames()

def returnNames():
    return ["Similarity", "Minimality", "Plausibility", "Validity", "Diversity"]