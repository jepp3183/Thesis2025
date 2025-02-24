import numpy as np
import pandas as pd
from lib.ext.baycon.common.ScoreCalculator import ScoreCalculator

def baseline_explainer(X, cluster_labels, classifier, cf_method, initial_point, target_cluster, binary=False):
    
    init_cluster = cluster_labels[initial_point]
    
    assert init_cluster != target_cluster, "Initial point is already in target cluster"

    if binary:
        X = X[(cluster_labels == init_cluster) | (cluster_labels == target_cluster)]
        cluster_labels = cluster_labels[(cluster_labels == init_cluster) | (cluster_labels == target_cluster)]

    classifier.fit(X, cluster_labels)
    print(f"Done training classifier. Score: {classifier.score(X, cluster_labels)}")

    initial_pred = classifier.predict(X[initial_point].reshape(1, -1))[0]
    assert initial_pred != target_cluster, "initial point classified in target cluster"

    cf = cf_method(X, classifier, initial_point, target_cluster)
    print(f"Found {len(cf)} counterfactuals")

    return cf