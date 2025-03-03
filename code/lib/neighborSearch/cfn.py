import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import json
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors

def neighborSearch(
        X,
        y,
        target,
        kmeans,
        instance_index=None,
        n=8,
        dis = lambda a,b : np.linalg.norm(a-b)):
    df = pd.DataFrame(np.column_stack((X, y)), columns=[f'x{i}' for i in range(X.shape[1])] + ['label'], dtype=float)
    predictor = lambda z : kmeans.predict(np.array([z]))

    pred_instance = 0.0
    if instance_index == None:
        instance_index = random.randint(0,len(df.values))
        new_instance = df.values[instance_index][:-1]
        pred_instance = predictor(new_instance)
        if pred_instance == 1.0:
            target = 0.0
        else:
            target = 1.0
    else:
        new_instance = df.values[instance_index][:-1]
        pred = predictor(new_instance)
        pred_instance = pred
        if pred == target:
            raise Exception("Faulty instance were given, target does not match")


    target_points = np.array((df[df["label"] == target]).values)[:,:-1]
    instance = df.values[instance_index][:-1]

    neigh = NearestNeighbors(n_neighbors=n)
    neigh.fit(target_points)
    neighbors = neigh.kneighbors([instance], n, return_distance=False)
    neighbors = target_points[neighbors[0]]
    
    counterfactuals = np.empty((0,len(instance)))
    for neighbor in neighbors:
        cf = instance.copy()
        changed_features = []

        while True:
            max_change = -float('inf')
            least_change = float('inf')
            max_feature = -1
            least_feature = -1

            for f in range(len(neighbor)):
                if f in changed_features:
                    continue
                temp_cf = cf.copy()
                temp_cf[f] = neighbor[f]
                f_dis = dis(temp_cf, neighbor)
                diff = np.abs(neighbor[f] - cf[f])
                score = f_dis / diff
                
                if predictor(temp_cf) != pred_instance:
                    if score < least_change:
                        least_change = score
                        least_feature = f
                    continue
                
                if score > max_change:
                    max_change = score
                    max_feature = f

            if max_feature == -1:
                cf[least_feature] = neighbor[least_feature]
                break
            else:
                changed_features.append(max_feature)
                cf[max_feature] = neighbor[max_feature]

        counterfactuals = np.append(counterfactuals, np.array([cf]), axis=0)

    return instance,counterfactuals