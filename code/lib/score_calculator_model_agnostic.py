import numpy as np

from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lib.ext.baycon.common.Target import Target

from mmd_critic import MMDCritic
from mmd_critic.kernels import RBFKernel

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

class ScoreCalculatorModelAgnostic:
    SCORE_JITTER = 0.95

    def __init__(self, initial_instance, initial_prediction, target, data_analyzer, base_calculator, amount_of_coreset_points, X, y):
        self._standard_score_calculator = base_calculator
        self._initial_instance = initial_instance
        self._initial_prediction = initial_prediction
        self._target = target
        self._data_analyzer = data_analyzer

        critic = MMDCritic(X, RBFKernel(sigma=1), criticism_kernel=RBFKernel(0.025), labels=y)

        protos, proto_labels = critic.select_prototypes(int(amount_of_coreset_points * 0.16))
        criticisms, criticism_labels = critic.select_criticisms(int(amount_of_coreset_points * 0.04), protos)
    
        stc_X = np.concatenate([protos, criticisms], axis=0)
        stc_y = np.concatenate([proto_labels, criticism_labels], axis=0)

        unlabeled_indices = np.random.rand(stc_y.shape[0]) < 0.3
        stc_y[unlabeled_indices] = -1

        etc = ExtraTreesClassifier()
        self._model = SelfTrainingClassifier(etc)
        self._model.fit(stc_X, stc_y)


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
        return self._model.predict_proba(instances)[:,self._target.target_value()]

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

