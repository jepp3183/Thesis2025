from enum import Enum
from lib.eval.eval_metrics import *
from tqdm import tqdm

class metrics(Enum):
    Similarity = 0,
    Sparsity = 1,
    Plausibility = 2,
    Validity = 3,
    Diversity = 4,
    Invalidation = 5,
    Correction = 6,
    Runtime = 7,
    PercentExplained = 8,
    ValidCFs = 9,
    
def run(
    method,
    centers,
    X,
    y,
    runtimes,
    m = [
        metrics.Similarity,
        metrics.Sparsity,
        metrics.Plausibility,
        metrics.Validity,
        metrics.Diversity,
        metrics.Invalidation,
        metrics.Correction,
        metrics.Runtime,
        metrics.PercentExplained,
        metrics.ValidCFs
    ],
    remove_invalid = True,
    dbg = False
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
            if len(cf_original) == 0:
                results.append([
                    [],
                    [],
                    [],
                    0.0,
                    [],
                    np.array([False for _ in range(len(cf_original))], dtype=bool),
                    np.array([False for _ in range(len(cf_original))], dtype=bool),
                    runtimes[i],
                    False,
                    0
                ])
                continue
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
                np.array([False for _ in range(len(cf_original))], dtype=bool),
                np.array([False for _ in range(len(cf_original))], dtype=bool),
                runtimes[i],
                False,
                0
            ])
            continue

        if metrics.Similarity in m:
            metric.append(cf_similarity(cf, instance))
        if metrics.Sparsity in m:
            metric.append(cf_minimality(cf, instance))
        if metrics.Plausibility in m:
            metric.append(cf_plausibility(cf, target, X, y))
        if metrics.Validity in m:
            metric.append(cf_validity(cf_original, target, centers))
        if metrics.Diversity in m:
            metric.append(cf_diversity(cf))
        if metrics.Invalidation in m:
            metric.append(cf_counterfactual_invalidation(cf_original, X, cf_data.instance, centers, target))
        if metrics.Correction in m:
            metric.append(cf_counterfactual_invalidation(cf_original, X, cf_data.instance, centers, target, correction=True))
        if metrics.Runtime in m:
            metric.append(runtimes[i])
        if metrics.PercentExplained in m:
            metric.append(cf_percent_explained(cf_original, target, centers))
        if metrics.ValidCFs in m:
            val = cf_validity(cf_original, target, centers)
            metric.append(val * len(cf_original))
        
        if dbg:
            print('\n'.join([f'{n}: {m}' for n,m in zip(returnNames(), metric)])) 

        results.append(metric)
    return results, returnNames()

def returnNames():
    return ["Similarity", "Sparsity", "Plausibility", "Validity", "Diversity", "Invalidation","Correction", "Runtime", "PercentExplained", "ValidCFs"]
