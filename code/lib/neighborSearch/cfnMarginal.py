import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors

# Counterfactual Neighbor search using greedy pick via marginal gain:
#   X                   - data points
#   y                   - labels for data points
#   target              - taarget cluster
#   model               - the clustering model, need to have predict function embed
#   instance_index      - instance for which counterfactuals are generated, is randomly set (for testing purposes) if not given
#   n                   - amount of counterfactuals generated, if higher than the size of target cluster this process fails
#   dis                 - distance metric
#
# returns the instance and a list of counterfactuals
#
def neighborSearchMarginal(
        X,
        y,
        target,
        model,
        instance_index=None,
        n=8,
        dis = lambda a,b : np.linalg.norm(a-b)):
    df = pd.DataFrame(np.column_stack((X, y)), columns=[f'x{i}' for i in range(X.shape[1])] + ['label'], dtype=float)
    predictor = lambda z : model.predict(np.array([z]))

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
    neighbors = np.array(target_points[neighbors[0]])
    
    center_orgin=np.array(model.cluster_centers_[int(pred_instance)])
    center_target=np.array(model.cluster_centers_[int(target)])

    counterfactuals = np.empty((0,len(instance)))
    for neighbor in neighbors:
        cf = np.array(instance.copy())
        changed_features = []

        while True:
            max_change = -float('inf')
            max_feature = -1

            if predictor(cf) == target:
                break

            for f in range(len(neighbor)):
                if f in changed_features:
                    continue
                temp_cf = cf.copy()
                temp_cf[f] = neighbor[f]

                score = marginalGain(temp_cf, cf, center_target, center_orgin, instance, dis, neighbor)
                
                if score > max_change:
                    max_change = score
                    max_feature = f

            changed_features.append(max_feature)
            cf[max_feature] = neighbor[max_feature]

        counterfactuals = np.append(counterfactuals, np.array([cf]), axis=0)

    return instance,counterfactuals

# cf_i, cf_(i-1), target_cluster, origin_cluster, instance, distance metric, neighbor
def marginalGain(point, past_point, t, o, x, dis, n):
    def cScore(point, target_center, origin_center):
      return dis(point, origin_center) / (dis(point, origin_center) + dis(point, target_center))

    gain = cScore(point,t,o) * (dis(x, n) - dis(n,point)) - cScore(past_point,t, o) * (dis(x, n) - dis(n,past_point))
    return max(gain,0) # Due to floating point inaccuracies