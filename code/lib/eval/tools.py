import numpy as np

def euclid_dis(x,y) -> float:
    return np.linalg.norm(x-y)

def center_prediction(p, centers, dis = euclid_dis) -> float:
    current_best_center = 0
    current_best_distance = float("inf")
    for i in range(len(centers)):
        distance = dis(p,centers[i])
        if distance < current_best_distance:
            current_best_center = i
            current_best_distance = distance
    return float(current_best_center)