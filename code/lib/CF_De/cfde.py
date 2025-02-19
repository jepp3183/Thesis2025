import numpy as np
import random

def Simple_CF_Descent(df, model, target, centers, instance_index=-1, stop_count=50, step_size=0.05, limit = 1000, dis = lambda a,b : euclid_dis(a,b)):
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
    while misses < stop_count and it < limit:
        y = target_points.sample().values[0][:-1]
        changes = []
        current_dis = dis(cf, instance)

        for i in range(len(y)):
            cf_prime = cf.copy()

            step = y[i] - cf_prime[i]
            
            cf_prime[i] += step * step_size

            distance_new = dis(cf_prime, instance)

            if distance_new < current_dis:
                changes.append((cf_prime,distance_new))

        if len(changes) == 0:
            misses += 1
        else:
            best = min(changes, key = lambda x: x[1])[0]
            if all([x == y for x,y in zip(cf,best)]):
                misses += 1
            else:    
                try:
                    prediction = model.predict([best])
                    if prediction[0] != int(target):
                        misses += 1
                        it += 1
                    else:
                        cf = best
                        misses = 0
                        history.append(best)
                except ValueError:
                    print("fail")
                    misses += 1
        it += 1

    if len(history) == 0:
        history.append((cf,0))
    print("Amount of changes: ", len(history))
    return instance,cf,history

def euclid_dis(x,y):
    return np.linalg.norm(x-y)