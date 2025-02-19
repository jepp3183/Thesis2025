import numpy as np

from lib.ext.baycon.common.Target import Target

ZERO_VALUE = 0.1


def score_y_away_from_target(min_value, turning_point, predictions, max_value):
    predictions_diff = np.abs(turning_point - predictions)
    total_diff = np.abs(max_value - min_value)
    result = (1 - np.divide(predictions_diff, total_diff)) * ZERO_VALUE
    return result


def score_y_reaching_target(min_value, turning_point, predictions, max_value):
    predictions_diff = np.abs(predictions - turning_point)
    total_diff = np.abs(max_value - min_value)
    normalized_scores = np.divide(predictions_diff, total_diff) * (1 - ZERO_VALUE) + ZERO_VALUE
    return normalized_scores

class ScoreCalculatorKmeans:
    SCORE_JITTER = 0.95

    def __init__(self, initial_instance, initial_prediction, target, data_analyzer, base_calculator, min_target_cluster_distance, max_target_cluster_distance, target_cluster_center):
        self._standard_score_calculator = base_calculator
        self._initial_instance = initial_instance
        self._initial_prediction = initial_prediction
        self._target = target
        self._data_analyzer = data_analyzer
        self._min_target_cluster_distance = min_target_cluster_distance
        self._max_target_cluster_distance = max_target_cluster_distance
        self._target_cluster_center = target_cluster_center

    def fitness_score(self, instances, predictions):
        # calculate closeness of the potential counterfactual to the initial instance.
        score_x = self.score_x(self._initial_instance, instances)
        score_y = self.score_y(instances)
        score_f = self.score_f(instances)
        # print(score_x,score_y,score_f)
        assert (score_x >= 0).all() and (score_y >= 0).all() and (score_f >= 0).all()
        fitness_score = score_x * score_y * score_f
        return np.round((fitness_score, score_x, score_y, score_f), 5)
    
    def score_y(self, instances):
        center_distances = self.euclidean_distance(instances)
        y_score = 1 - ((center_distances - self._min_target_cluster_distance)/(self._max_target_cluster_distance - self._min_target_cluster_distance))
        y_score[y_score < 0] = 0
        y_score[y_score > 1] = 1
        return y_score
        

    def euclidean_distance(self, y):
        prediction_distances = [np.linalg.norm(i-self._target_cluster_center) for i in y]
        return prediction_distances

    def gower_distance(self, origin_instance, instances):
        return self._standard_score_calculator.gower_distance(origin_instance, instances)

    def score_x(self, from_instance, to_instances):
        return self._standard_score_calculator.score_x(from_instance, to_instances)
    
    def score_f(self, instances):
        return self._standard_score_calculator.score_f(instances)

    def filter_instances_within_score(self, instance_from, instances_to_filter):
        return self._standard_score_calculator.filter_instances_within_score(instance_from, instances_to_filter)
    
    def near_score(self, score, scores_to_check):
        return self._standard_score_calculator.near_score(score, scores_to_check)

