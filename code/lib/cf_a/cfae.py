import numpy as np
import random
import pandas as pd

def CF_Descent(
        X,
        y, 
        target, 
        centers, 
        model=None, 
        instance_index=-1, 
        stop_count=100, 
        step_size=0.05, 
        limit = 10000,
        feature_penalty = 1.001, 
        dis = lambda a,b : euclid_dis(a,b),
        immutable_features = []):
    
    df = pd.DataFrame(np.column_stack((X, y)), columns=[f'x{i}' for i in range(X.shape[1])] + ['label'], dtype=float)
    return Simple_CF_Ascent(df, target, centers, model, instance_index, stop_count, step_size, limit, feature_penalty, dis, immutable_features)


def Simple_CF_Ascent(
        df,
        target, 
        centers, 
        model=None, 
        instance_index=-1, 
        stop_count=100, 
        step_size=0.05, 
        limit = 10000,
        feature_penalty = 1.001, 
        dis = lambda a,b : euclid_dis(a,b),
        immutable_features = []):

    predictor = None
    if model == None:
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

    pred_instance = 0.0
    if instance_index == -1:
        while True:
            index = random.randint(0,len(df.values))
            new_instance = df.values[index][:-1]
            pred = predictor(new_instance)
            if pred != target:
                print("Generation counterfacutal from cluster: " + str(pred) + " , Into cluster: " + str(target))
                instance_index = index
                break


    history = []
    misses = 0
    instance = df.values[instance_index][:-1]

    cf = centers[int(target)].copy()

    target_center = centers[int(target)].copy()

    it = 0
    changed_features = []
    while misses < stop_count and it < limit:
        y = target_points.sample().values[0][:-1]
        changes = []
        current_dis = dis(cf, instance)

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

            distance_new = dis(cf_prime, target_center) * penalty

            if distance_new > current_dis:
                changes.append((cf_prime,distance_new, i))

        if len(changes) == 0:
            misses += 1
        else:
            best = max(changes, key = lambda x: x[1])
            f = best[2]
            best = best[0]
            if all([x == y for x,y in zip(cf,best)]):
                misses += 1
            else:    
                try:
                    prediction = predictor(best)
                    if prediction == target:
                        break
                    elif prediction == pred_instance:
                        cf = best
                        misses = 0
                        history.append(best)
                        if f not in changed_features:
                            changed_features.append(f)
                    else:
                        misses += 1
                except ValueError:
                    print("fail")
                    misses += 1
        it += 1

    a_cha = len(history)
    if len(history) == 0:
        history.append(cf)
        a_cha = 0
    print("Amount of changes: ", a_cha)
    print("Number of changed features:",len(changed_features))
    return instance,cf,history

def euclid_dis(x,y):
    return np.linalg.norm(x-y)