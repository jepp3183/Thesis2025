import numpy as np
import random

def Simple_CF_Descent(
        df, 
        target, 
        centers, 
        model=None, 
        instance_index=-1, 
        stop_count=100, 
        step_size=0.05, 
        limit = 10000,
        feature_penalty = 1.001, 
        dis = lambda a,b : euclid_dis(a,b)):

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

    cf = centers[int(target)]
    target_points = df[df["label"] == target]

    if instance_index == -1:
        while True:
            index = random.randint(0,len(df.values))
            new_instance = df.values[index]
            if new_instance[-1] != target:
                instance_index = index
                break


    history = []
    misses = 0
    instance = df.values[instance_index]
    if instance[-1] == target:
        raise Exception("cannot create CF for current class")
    instance = instance[:-1]

    it = 0
    changed_features = []
    while misses < stop_count and it < limit:
        y = target_points.sample().values[0][:-1]
        changes = []
        current_dis = dis(cf, instance)

        for i in range(len(y)):
            penalty = 1.0
            if i not in changed_features:
                if len(changed_features) == 0:
                    penalty = 1.0
                else:
                    penalty = np.pow([feature_penalty], [len(changed_features)])[0]

            cf_prime = cf.copy()

            step = y[i] - cf_prime[i]
            
            cf_prime[i] += step * step_size

            distance_new = dis(cf_prime, instance) * penalty

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
                    if prediction != int(target):
                        misses += 1
                        it += 1
                    else:
                        cf = best
                        misses = 0
                        history.append(best)
                        if f not in changed_features:
                            changed_features.append(f)
                except ValueError:
                    print("fail")
                    misses += 1
        it += 1

    if len(history) == 0:
        history.append((cf,0))
    print("Amount of changes: ", len(history))
    print("Number of changed features:",len(changed_features))
    return instance,cf,history

def euclid_dis(x,y):
    return np.linalg.norm(x-y)