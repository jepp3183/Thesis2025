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

def neighborSearchMarginal(
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


    def cScore(point, target_center, origin_center):
        return dis(point, origin_center) / (dis(point, origin_center) + dis(point, target_center))

    target_points = np.array((df[df["label"] == target]).values)[:,:-1]
    instance = df.values[instance_index][:-1]

    neigh = NearestNeighbors(n_neighbors=n)
    neigh.fit(target_points)
    neighbors = neigh.kneighbors([instance], n, return_distance=False)
    neighbors = target_points[neighbors[0]]
    
    
    center_orgin=kmeans.cluster_centers_[int(pred_instance)]
    center_target=kmeans.cluster_centers_[int(target)]

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

                def marginalGain(point, past_point, t, o, x):
                    gain = cScore(point,t,o) * (dis(x, neighbor) - dis(neighbor,point)) - cScore(past_point,t, o) * (dis(x, neighbor) - dis(neighbor,past_point))
                    return max(gain,0)

                score = marginalGain(temp_cf, cf, center_target, center_orgin, instance)
                
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