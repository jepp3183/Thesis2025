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
        dis = lambda a,b : np.linalg.norm(a-b),
        marginalGainBool = True):
    df = pd.DataFrame(np.column_stack((X, y)), columns=[f'x{i}' for i in range(X.shape[1])] + ['label'], dtype=float)
    predictor = lambda z : model.predict(np.array([z]),)

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
        if marginalGainBool:
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
        else:
            picks = []
            for f in range(len(neighbor)):
                temp_cf = cf.copy()
                temp_cf[f] = neighbor[f]

                new_score = cScore(temp_cf,instance,center_target,center_orgin)*(dis(instance, neighbor)**2 - dis(neighbor,temp_cf)**2)
                picks.append((new_score, f))
                    
            picks.sort(reverse=True, key=lambda x: x[0])

            #print(f"picks: {picks}")
            for _,index in picks:
                cf[index] = neighbor[index]
                if predictor(cf) == target:
                    break

        counterfactuals = np.append(counterfactuals, np.array([cf]), axis=0)

    return instance,counterfactuals


def cScore(cf, x, target_center, origin_center):
        V = target_center - x
        change = cf - x

                #print(f"V: {V}, cf: {cf}, x: {x}, change: {change}, target_center: {target_center} dot: {np.dot(change, V)}")
        #xPrimeV = np.dot(change, V) / np.linalg.norm(V)
        change = change
        xPrimeV = np.dot(V, change) / np.linalg.norm(V)**2
        return xPrimeV
        #xV = np.dot(x, V)
        #xV = xV / np.linalg.norm(V)

        #xPrimeV = 1 - (1 / (1 + (dis(cf, origin_center) / dis(cf, target_center))))
        
        #return xPrimeV + np.linalg.norm(x-n)# - np.min(xV, 0)
        #if (dis(x,target_center)**2 - dis(cf, target_center)**2) + (dis(x,n)**2-dis(cf, n)**2) < 0.0:
        #    print(f"first: {dis(x,target_center)**2 }, 2nd: {dis(cf, target_center)**2}, third: {dis(x,n)**2}, fourth: {dis(cf, n)**2}")
        #return (dis(x,target_center) - dis(cf, target_center)) + (dis(x,n)-dis(cf, n))

        #return dis(cf, origin_center) / (dis(cf, origin_center) + dis(cf, target_center))

# cf_i, cf_(i-1), target_cluster, origin_cluster, instance, distance metric, neighbor
def marginalGain(cf, past_point, t, o, x, dis, n):


    gain = cScore(cf,x,t,o) * (dis(x, n) - dis(n,cf)) - cScore(past_point,x,t, o) * (dis(x, n) - dis(n,past_point))
    #gain = cScore(cf,x,t,o) - cScore(past_point,x,t, o)
    #gain = (dis(x, n)**2 - dis(n,cf)**2) - (dis(x, n)**2 - dis(n,past_point)**2)
    #if gain < 0.0:
        #print(cScore(point,x,t,o) - cScore(x,x,t, o))
        
    #    print(f"Gain: {gain}, cScore: {cScore(cf,x,t,o)}, past: {cScore(past_point,x,t,o)}, first: {(dis(x, n) - dis(n,cf))}, second: {(dis(x, n) - dis(n,past_point))}")
    #    print(f"score-percent-change: {cScore(cf,x,t,o) / cScore(past_point,x,t,o)}, dis-percent-change: {(dis(x, n) - dis(n,cf)) / (dis(x, n) - dis(n,past_point))}")
    #assert gain >= 0.0, "Gain should be non-negative"
    return gain