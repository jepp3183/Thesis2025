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
        immutable_features = [],
        new_immutable_ratio = 0.35):
    
    df = pd.DataFrame(np.column_stack((X, y)), columns=[f'x{i}' for i in range(X.shape[1])] + ['label'], dtype=float)
    return Simple_CF_Descent(df, target, centers, model, instance_index, stop_count, step_size, limit, feature_penalty, dis, immutable_features, new_immutable_ratio)


#TODO start by greedily setting features to the instance choosing features that change the distance the least, then do descent afterwards!!!
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
        dis = lambda a,b : euclid_dis(a,b),
        immutable_features = [],
        new_immutable_ratio = 0.35):

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

    cf = centers[int(target)].copy()
    target_points = df[df["label"] == target]

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

    # taking care of immutables
    for f in immutable_features:
        cf[f] = instance[f].copy()

    #First we need to reset some features to obtain minimality
    found_starting_point = False
    temp_imuts = immutable_features.copy()
    while found_starting_point == False:
        
        if float(len(temp_imuts))/float(len(instance)) > new_immutable_ratio:
            break

        smallest_distance = float('inf')
        best_feature = -1
        for f in range(len(instance)):
            if f in temp_imuts:
                continue

            cf_prime = cf.copy()

            cf_prime[f] = instance[f].copy()
            distance_new = dis(cf_prime, instance)

            if distance_new < smallest_distance:
                pre_class = predictor(cf_prime)
                if pre_class == target:
                    smallest_distance = distance_new
                    best_feature = f
                
        if best_feature == -1:
            found_starting_point = True
        else:
            cf[best_feature] = instance[best_feature].copy()
            temp_imuts.append(best_feature)

    immutable_features = temp_imuts
    print("Features that can be changed count: ", len(instance) - len(immutable_features))

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
                    if prediction != target:
                        misses += 1
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

    a_cha = len(history)
    if len(history) == 0:
        history.append(cf)
        a_cha = 0
    print("Amount of changes: ", a_cha)
    print("Number of changed features:",len(changed_features))
    return instance,cf,history

def euclid_dis(x,y):
    return np.linalg.norm(x-y)