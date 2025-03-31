import numpy as np
import torch


def random_opt(start, gain, max_fails=25):
    start = torch.from_numpy(start)
    fails = 0
    iter = 0
    history = start
    # print(start)
    best = start
    best_gain = gain(start)
    while True:
        step = torch.normal(0, 0.1, start.shape)
        # print(step)
        cand = best + step
        cand_gain = gain(cand)
        if cand_gain > best_gain:
            best = cand
            best_gain = cand_gain
            history = torch.vstack([history, best])
        else: 
            fails += 1
        iter += 1
        if fails >= max_fails:
            break
    # print(f"best: {best}, best_gain: {best_gain}") 
    # print(f"hist shape: {torch.array(history).shape}")
    print(f"iter: {iter}")
    return best, history


def gradient_ascent(start: np.ndarray, gain, lr = 0.1, dbg=False, max_iter=1000):
    current = torch.from_numpy(start)
    current.requires_grad = True

    iter = 1
    imp = float("inf")
    fails = 0
    best = current
    best_score = 0
    history = current
    # while (torch.linalg.norm(grad) > 0.001 or imp > 0.01) and iter < 2000:
    while iter < max_iter and fails < 100:
        foo = gain(current)
        foo.backward()
        grad = current.grad
        if grad is None:
            print("grad is None")
            break
        grad[grad.isnan()] = 0

        with torch.no_grad():
            prev = gain(current)
            damp = iter // 100 + 1
            current = current + (lr / damp) * grad
            score = gain(current)
            if score > best_score:
                best = current
                best_score = score
                fails = 0
            else:
                fails += 1
            imp = score - prev
            
        current.requires_grad = True
        
        history = torch.vstack([history, current])
        iter += 1
        if dbg:
            print(f"iter: {iter}, score: {score}, imp: {imp}, grad: {torch.linalg.norm(grad)}")
    # print(f"best: {best}, best_gain: {best_gain}") 
    # print(f"hist shape: {torch.array(history).shape}")
    print(f"iter: {iter}, score: {best_score}")
    return best.detach().numpy(), history.detach().numpy()
    
    

class Gainer:
    def __init__(self, C, X, target, x, **kwargs):
        """
        Parameters
        ----------
        C: cluster centers

        X: data

        target: target cluster label

        x: instance to be explained
        """
        print(f"X: {X.shape}")
        print(f"C: {C.shape}")
        print(f"x: {x.shape}")

        self.eps = kwargs.get("eps", 0)

        self.C = torch.from_numpy(C)
        self.X = torch.from_numpy(X)
        self.target = target
        self.x = torch.from_numpy(x)

        self.instance_cluster = self._classify(self.x)[0]
        assert self.instance_cluster != self.target

        self.y = self._classify(self.X)
        self.x_idx = torch.where(torch.all(self.X == self.x, 1))[0][0]

        self.max_t, self.min_t = self._find_cluster_distances()
        self.mean_t = torch.mean(
            torch.linalg.norm(self.X[self.y==self.target] - self.C[[self.target]], axis=1)
        )

        feature_mins = self.X.min(0).values
        feature_maxs = self.X.max(0).values
        self.feature_ranges = feature_maxs - feature_mins

        self.gain_weights = {
            # self.ygain: 1,
            # self.sim_gain: 1,
            # self.dist_gain: 0.5,
            self.sigmoid_hinge_gain: 1,
            # self.is_valid: 1,
            # self.sparsity_gain: 1,
            self.gower_gain: 1,
            # self.baycon_gain: 1
            self.smooth_is_valid: 1
        }

    def _classify(self, X):
        dists = torch.linalg.norm(X[:, None] - self.C, dim=2)
        return torch.argmin(dists, 1) 

    def _find_cluster_distances(self):
        c = self.C[[self.target]]
        dists = torch.linalg.norm(self.X[self.y==self.target] - c, axis=1)
        return torch.max(dists), torch.min(dists)
    
    def ygain(self, cf):
        d = torch.linalg.norm(cf - self.C[[self.target]])
        y = (d - self.min_t)/(self.max_t - self.min_t)
        return 1 - torch.clip(y, 0, 1)
    
    def sim_gain(self, cf):
        d = torch.linalg.norm(cf - self.x)
        d_sim = (d - self.min_t)/(self.max_t - self.min_t)
        return 1 - torch.clip(d_sim, 0, 1)

    def dist_gain(self, cf):
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf)
        d = torch.linalg.norm(cf - self.x)
        return 1 - d
    
    def is_valid(self, cf):
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf)
        cf_cluster = self._classify(cf)
        valid = (cf_cluster == self.target)
        ret = torch.maximum(valid, torch.tensor(0.5))
        return ret[0]

    def smooth_is_valid(self, cf):
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf)


        mask = torch.arange(self.C.size(0)) != self.target
        baz = self.C[mask]
        centers = baz.reshape(self.C.shape[0] - 1, self.C.shape[1])


        dists = torch.linalg.norm(cf - centers, dim = 1)

        min_dist = torch.min(dists)
        t_dist = torch.linalg.norm(cf - self.C[[self.target]])
        
        # return min_dist / (t_dist + min_dist)
        foo = (min_dist + t_dist) / (2 * t_dist)
        # print everything above
        bar = torch.minimum(torch.tensor(1), foo)
        return bar

    def sig(self, d):
        off = (1 - self.eps) * self.max_t
        # off = 0.5
        base = 10000
        e = base ** (d - off)
        return 1 / (1 + e)
        

    def sigmoid_hinge_gain(self, cf):
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf)
        d = torch.linalg.norm(cf - self.C[[self.target]])
        return self.sig(d)
        
    def sparsity_gain(self, cf):
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf)

        i = torch.isclose(cf, self.x, atol=0.01).mean(dtype=torch.float64)
        return torch.maximum(torch.tensor(0.1 / cf.shape[1]), i)

    def gower_gain(self, cf):
        """
        Inverse Gower's distance between the counterfactual and the original instance.
        WARNING: CAUSES NaNs in gradient!
        """
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf)
        diffs = torch.abs(cf - self.x)
        scaled_diffs = diffs / self.feature_ranges
        scaled_diffs[scaled_diffs != scaled_diffs] = 0
        sims = 1 - scaled_diffs
        res = torch.mean(sims)
        return res
    
    def baycon_gain(self, cf):
        return self.sim_gain(cf) * self.sparsity_gain(cf) * self.gower_gain(cf)
        
    def gain(self, cf):
        # gain = sum([term(cf)*weight for term,weight in self.gain_weights.items()])
        # gain = math.prod([(term(cf)) for term in self.gain_weights.keys()])
        gain = torch.tensor(1, dtype=torch.float64)
        for term in self.gain_weights.keys():
            t = term(cf)
            # print(f"{term.__name__}: {t}")
            gain *= t
        return gain
