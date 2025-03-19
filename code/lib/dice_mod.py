import numpy as np
import math

def random_opt(start, gain, max_fails=50):
    fails = 0
    iter = 0
    history = [start[0]]
    print(start)
    best = start
    best_gain = gain(start)
    while True:
        step = np.random.normal(0, 1, start.shape)
        # print(step)
        cand = best + step
        cand_gain = gain(cand)
        if cand_gain > best_gain:
            best = cand
            best_gain = cand_gain
            history.append(best[0])
        else: 
            fails += 1
        iter += 1
        if fails >= max_fails:
            break
    print(f"best: {best}, best_gain: {best_gain}") 
    print(f"hist shape: {np.array(history).shape}")
    print(f"iter: {iter}")
    return best, np.array(history)
    

class Gainer:
    def __init__(self, C, X, target, x):
        """
        Parameters
        ----------
        C: cluster centers

        X: data

        target: target cluster label

        x: instance to be explained
        """
        self.C = C
        self.X = X
        self.target = target
        self.x = x

        self.instance_cluster = self._classify(x)[0]
        assert self.instance_cluster != self.target

        self.y = self._classify(X)
        self.x_idx = np.where(np.all(X == x, axis=1))[0][0]

        self.max_t, self.min_t = self._find_cluster_distances()
        self.mean_t = np.mean(
            np.linalg.norm(self.X[self.y==self.target] - self.C[[self.target]], axis=1)
        )

        feature_mins = X.min(axis=0)
        feature_maxs = X.max(axis=0)
        self.feature_ranges = feature_maxs - feature_mins

        self.loss_weights = {
            # self.ygain: 1,
            # self.sim_gain: 1,
            self.sigmoid_hinge_gain: 1,
            self.is_valid: 1,
            # self.sparsity_gain: 1,
            self.gower_gain: 0.3,
            # self.baycon_gain: 1
        }

    def _classify(self, X):
        dists = np.linalg.norm(X[:, None] - self.C, axis=2)
        return np.argmin(dists, axis=1) 

    def _find_cluster_distances(self):
        c = self.C[[self.target]]
        dists = np.linalg.norm(self.X[self.y==self.target] - c, axis=1)
        return np.max(dists), np.min(dists)
    
    def ygain(self, cf):
        d = np.linalg.norm(cf - self.C[[self.target]])
        yloss = (d - self.min_t)/(self.max_t - self.min_t)
        return np.clip(yloss, 0, 1)
    
    def sim_gain(self, cf):
        d = np.linalg.norm(cf - self.x)
        d_sim = (d - self.min_t)/(self.max_t - self.min_t)
        return 1 - np.clip(d_sim, 0, 1)
    
    # def is_closer(self, cf):
    #     """Returns true if the given cf is closer to target center than original center"""
    #     return np.linalg.norm(cf - self.C[[self.instance_cluster]]) > np.linalg.norm(cf - self.C[[self.target]])
    
    # def binary_hinge_loss(self, cf):
    #     return int(not self.is_closer(cf))
    
    # def hinge_similarity_loss(self, cf):
    #     halfway = np.mean([self.C[[self.target]], self.C[[self.instance_cluster]]], axis=0)
    #     d = np.linalg.norm(cf - halfway)
    #     d_sim = d / np.linalg.norm(self.C[[self.target]] - halfway[:, None])

    #     return int(not self.is_closer(cf)) + d_sim - 1

    def is_valid(self, cf):
        cf_cluster = self._classify(cf)
        valid = int((cf_cluster == self.target)[0])
        return max(valid, 0.1)
    
    def sigmoid_hinge_gain(self, cf):
        d = np.linalg.norm(cf - self.C[[self.target]])
        off = self.max_t
        return 1 / (1 + np.exp(d - off))
        
    def sparsity_gain(self, cf):
        return np.isclose(cf, self.x, atol=0.1).mean()

    def gower_gain(self, cf):
        diffs = np.abs(cf - self.x)
        scaled_diffs = diffs / self.feature_ranges
        sims = 1 - scaled_diffs
        return np.mean(sims)
    
    def baycon_gain(self, cf):
        return self.sim_gain(cf) * self.sparsity_gain(cf) * self.gower_gain(cf)
        
    def gain(self, cf):
        l = sum([term(cf)*weight for term,weight in self.loss_weights.items()])
        # l = math.prod([(term(cf)) for term in self.loss_weights.keys()])
        return l
