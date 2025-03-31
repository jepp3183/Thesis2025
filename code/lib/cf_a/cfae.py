import numpy as np
import random
import pandas as pd

# Counterfactual ascent takes the following parameters:
#   X                   - data points
#   y                   - labels for data points
#   target              - taarget cluster
#   centers             - centers for clusters
#   model               - the clustering model, if not set euclidian distance is used instead
#   instance_index      - instance for which counterfactuals are generated, is randomly set (for testing purposes) if not given
#   stop_count          - after this many iterations without improvements stop, too low could result in non-valid counterfactuals
#   step_size           - how much each feature is changed in each iteration, too slow does not converge but too fast given solution with low similarity
#   limit               - the maximum amount of iterations allowed
#   feature_penalty     - punishes the algorithm from changing too many features, very sensitive to small changes
#   dis                 - distance metric
#   immutable_features  - list of which features are immutable as indexes, f.eks. [2,8,11] for feature 2,8,11
#   center_mode         - decides whether the ascent target is a random point in target cluster or the center, 
#                         random points in general work better hence why it's false by default
#
#
# Returns the instance, the resulting counterfactual and the history of changes
#
def CF_Ascent(
        X,
        y, 
        target, 
        centers, 
        model=None, 
        instance_index=None, 
        stop_count=100, 
        step_size=0.05, 
        limit = 10000,
        feature_penalty = 1.001, 
        dis = lambda a,b : euclid_dis(a,b),
        immutable_features = [],
        center_mode = False):
    
    df = pd.DataFrame(np.column_stack((X, y)), columns=[f'x{i}' for i in range(X.shape[1])] + ['label'], dtype=float)
    return Simple_CF_Ascent(df, target, centers, model, instance_index, stop_count, step_size, limit, feature_penalty, dis, immutable_features, center_mode)

# This constructor exists for test purposes, refer to above constructor for use outside this sub-directory
def Simple_CF_Ascent(
        df,
        target, 
        centers, 
        model=None, 
        instance_index=None, 
        stop_count=100, 
        step_size=0.05, 
        limit = 10000,
        feature_penalty = 1.001, 
        dis = lambda a,b : euclid_dis(a,b),
        immutable_features = [],
        center_mode = True):

    predictor = None
    if model == None: # If model has not been provided use euclidian distance
        def center_prediction(p):
            current_best_center = 0
            current_best_distance = float("inf")
            for i in range(len(centers)):
                distance = dis(p,centers[i])
                if distance < current_best_distance:
                    current_best_center = i
                    current_best_distance = distance
            return float(current_best_center)

        predictor = lambda p : center_prediction(p)
    else:
        predictor = lambda p : model.predict([p])[0]

    target_points = df[df["label"] == target]

    if instance_index == None: # Only for testing, pick random point and target
        while True:
            index = random.randint(0,len(df.values))
            new_instance = df.values[index][:-1]
            pred = predictor(new_instance)
            if pred != target:
                instance_index = index
                break
    else:
        new_instance = df.values[instance_index][:-1]
        pred = predictor(new_instance)
        if pred == target:
            raise Exception("Faulty instance were given, target does not match")

    history = []
    misses = 0
    instance = np.array(df.values[instance_index][:-1])

    cf = np.array(instance.copy())

    target_metric = np.array(centers[int(target)].copy())
    if (center_mode == False): # is false by default
        target_metric = np.array(target_points.sample().values[0][:-1])

    it = 0
    changed_features = []
    while misses < stop_count and it < limit:
        y = np.array(target_points.sample().values[0][:-1])
        changes = []

        current_dis = dis(cf, target_metric)

        for i in range(len(y)):
            if i in immutable_features:
                continue
            penalty = 1.0
            if i not in changed_features:
                if len(changed_features) == 0:
                    penalty = 1.0
                else:
                    penalty = np.pow([feature_penalty], [len(changed_features)])[0]

            cf_prime = cf.copy()

            step = y[i] - cf_prime[i]
            
            cf_prime[i] += step * step_size

            distance_new = dis(cf_prime, target_metric) * penalty

            if distance_new < current_dis:
                changes.append((cf_prime,distance_new, i))

        if len(changes) == 0:
            misses += 1
        else:
            best = min(changes, key = lambda x: x[1])
            f = best[2]
            best = best[0]
            if all([x == y for x,y in zip(cf,best)]):
                misses += 1
            else:    
                try:
                    prediction = predictor(best)
                    if prediction == target:
                        cf = best
                        break
                    else:
                        cf = best
                        misses = 0
                        history.append(best)
                        if f not in changed_features:
                            changed_features.append(f)
                except ValueError:
                    misses += 1
        it += 1

    if len(history) == 0:
        history.append(cf)
        
    return instance,cf,np.array(history)

def euclid_dis(x,y):
    return np.linalg.norm(x-y)