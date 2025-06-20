import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA


def random_opt(start, gain, max_fails=100):
    """
    Random optimization algorithm. 
    Picks a random point close to current, and moves to it if it improves the objective.
    Stops after max_fails failed attempts.
    """
    start = torch.from_numpy(start)
    fails = 0
    iter = 0
    history = start
    best = start
    best_gain = gain(start)
    while True:
        step = torch.normal(0, 0.1, start.shape)
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
    print(f"iter: {iter}, score: {best_gain}")
    return best, history


def gradient_ascent(start: np.ndarray, gain, lr=0.1, dbg=False, max_iter=1000):
    """
    Gradient ascent algorithm.
    Very basic implementation for testing. Too dependent on parameters
    """
    current = torch.from_numpy(start)
    current.requires_grad = True

    iter_count = 1
    fails = 0
    best = current
    best_score = 0
    history = current
    
    while iter_count < max_iter and fails < 100:
        score = gain(current)
        score.backward()
        grad = current.grad
        if grad is None: 
            print("None gradient")
            break
        grad[grad.isnan()] = 0

        with torch.no_grad():
            prev_score = gain(current)
            
            damp = iter_count // 100 + 1
            current = current + (lr / damp) * grad
            
            new_score = gain(current)
            improvement = new_score - prev_score
            
            if new_score > best_score:
                best = current
                best_score = new_score
                fails = 0
            else:
                fails += 1
        current.requires_grad = True
        
        history = torch.vstack([history, current])
        if dbg:
            print(f"iter: {iter_count}, score: {new_score}, imp: {improvement}, grad: {torch.linalg.norm(grad)}")
            
        iter_count += 1
        
    print(f"iter: {iter_count}, score: {best_score}")
    return best.detach().numpy(), history.detach().numpy()

def adam_opt(start: np.ndarray, gain, lr=0.1, max_iter=1000):
    """
    ADAM optimization algorithm using PyTorch's optimizer.
    """
    current = torch.from_numpy(start).clone().detach()
    current.requires_grad = True
    
    optimizer = torch.optim.Adam([current], lr=lr)
    
    iter_count = 1
    fails = 0
    best = current.clone().detach()
    best_score = gain(best)
    history = current.clone().detach()
    
    while iter_count < max_iter and fails < 100:
        # ADAM minimizes, so negate
        optimizer.zero_grad()
        score = -gain(current)
        score.backward()
        if current.grad is None: 
            print("None gradient")
            break
        current.grad[torch.isnan(current.grad)] = 0
        
        optimizer.step()
        
        with torch.no_grad():
            new_score = gain(current)
            
            if new_score > best_score:
                best = current.clone().detach()
                best_score = new_score
                fails = 0
            else:
                fails += 1
            
            history = torch.vstack([history, current.clone().detach()])
        
        current.requires_grad = True
        
        iter_count += 1
    
    print(f"iter: {iter_count}, score: {best_score}")
    return best.detach().numpy(), history.detach().numpy()


class Gainer:
    def __init__(self, C, X, target, x, **kwargs):
        """
        Class for computing the score of a counterfactual solution.
        The score is the product of several terms, which are defined in the gain_weights dictionary.

        Parameters
        ----------
        C: cluster centers

        X: data

        target: target cluster label

        x: instance to be explained

        kwargs:
            eps [0-1], default: 0. Controls how close the solution will be to the target cluster center.
        """
        # print(f"X: {X.shape}")
        # print(f"C: {C.shape}")
        # print(f"x: {x.shape}")

        self.eps = kwargs.get("eps", 0)

        self.C = torch.from_numpy(C)
        self.X = torch.from_numpy(X)
        self.target = target
        self.x = torch.from_numpy(x)

        self.instance_cluster = self._classify(self.x)[0]
        assert self.instance_cluster != self.target, "Instance is already in target cluster"

        self.y = self._classify(self.X)
        self.x_idx = torch.where(torch.all(self.X == self.x, 1))[0][0]

        self.max_t, self.min_t = self._find_cluster_distances()
        self.mean_t = torch.mean(
            torch.linalg.norm(self.X[self.y==self.target] - self.C[[self.target]], axis=1)
        )

        dists = torch.linalg.norm(self.C - self.C[self.target], axis=1)
        sorted, _indices = torch.sort(dists, descending=False)
        decision_len = sorted[1] / 2
        self.sig_offset = min(decision_len, self.max_t)

        feature_mins = self.X.min(0).values
        feature_maxs = self.X.max(0).values
        self.feature_ranges = feature_maxs - feature_mins

        self.gain_weights = {
            self.gower_gain: 1,
            self.sigmoid_hinge_gain: 1,
            # self.smooth_is_valid: 1,
            # self.baycon_gain: 1,
            # self.dist_gain: 0.5,
            # self.is_valid: 1,
            # self.sim_gain: 1,
            # self.sparsity_gain: 1,
            # self.ygain: 1
        }

    def _classify(self, X):
        dists = torch.linalg.norm(X[:, None] - self.C, dim=2)
        return torch.argmin(dists, 1) 

    def _find_cluster_distances(self):
        c = self.C[[self.target]]
        dists = torch.linalg.norm(self.X[self.y==self.target] - c, axis=1)
        return torch.max(dists), torch.min(dists)

    def gain(self, cf):
        # gain = sum([term(cf)*weight for term,weight in self.gain_weights.items()])
        # gain = math.prod([(term(cf)) for term in self.gain_weights.keys()])
        gain = torch.tensor(1, dtype=torch.float64)
        # s = []
        for term in self.gain_weights.keys():
            t = term(cf)
            # s.append(f"{term.__name__}: {t}")
            gain *= t
        # print(f"score: {gain}, gains: {s}")
        return gain
    
    def gower_gain(self, cf):
        """
        Gower's similarity between the counterfactual and the original instance.
        """
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf)
        diffs = torch.abs(cf - self.x)
        scaled_diffs = diffs / self.feature_ranges
        scaled_diffs[scaled_diffs != scaled_diffs] = 0
        sims = 1 - scaled_diffs
        res = torch.mean(sims)
        # assert 0 <= res and res <= 1, f"Invalid Gower gain: {res}"
        return res
    
    def sig(self, d):
        off = (1 - self.eps)*self.sig_offset

        if d < off:
            base = 2.718
            e = base ** (50*(d - off))
            res = 1 / (1 + e)
            return res
            
        d_instance_target = torch.linalg.norm(self.C[[self.target]] - self.x)
        res = (d_instance_target - d)/((d_instance_target - off)*2)

        return torch.maximum(torch.tensor(0.), res)
        
    def sigmoid_hinge_gain(self, cf):
        """
        Sigmoid-like function that encourages being close to the target cluster center.
        """
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf)
        d = torch.linalg.norm(cf - self.C[[self.target]])
        return self.sig(d)
    
    def smooth_is_valid(self, cf):
        """
        Smoothed version of is_valid that encourages the counterfactual to be in the target cluster.
        """
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf)


        mask = torch.arange(self.C.size(0)) != self.target
        baz = self.C[mask]
        centers = baz.reshape(self.C.shape[0] - 1, self.C.shape[1])


        dists = torch.linalg.norm(cf - centers, dim = 1)

        min_dist = torch.min(dists)
        t_dist = torch.linalg.norm(cf - self.C[[self.target]])
        
        foo = (min_dist + t_dist) / (2 * t_dist)
        bar = torch.minimum(torch.tensor(1), foo)
        return bar
    

    # Below are some alternative functions used in testing, but not in the final implementation

    def baycon_gain(self, cf):
        return self.sim_gain(cf) * self.sparsity_gain(cf) * self.gower_gain(cf)

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

    def sim_gain(self, cf):
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf)
        d = torch.linalg.norm(cf - self.x)
        d_sim = (d - self.min_t)/(self.max_t - self.min_t)
        return 1 - torch.clip(d_sim, 0, 1)

    def sparsity_gain(self, cf):
        if type(cf) is np.ndarray:
            cf = torch.from_numpy(cf)

        i = torch.isclose(cf, self.x, atol=0.0001).mean(dtype=torch.float64)
        return torch.maximum(torch.tensor(0.1 / cf.shape[1]), i)

    def ygain(self, cf):
        d = torch.linalg.norm(cf - self.C[[self.target]])
        y = (d - self.min_t)/(self.max_t - self.min_t)
        return 1 - torch.clip(y, 0, 1)


def plot_heatmap(X, y, C, random_point, fn, target_cluster, use_pca=False, solutions=None, histories=None, ax=None, heatmap=True):
    if use_pca:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        C = pca.transform(C)
        random_point = pca.transform(random_point)
        if solutions is not None:
            solutions = [pca.transform(solution) for solution in solutions]
        if histories is not None:
            histories = [pca.transform(history) for history in histories]
    
    p = plt if ax is None else ax


    # Create a grid for the heatmap
    if not use_pca and heatmap:
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        values = np.array([fn(point.reshape(1, -1)) for point in grid_points])
        grid = values.reshape(xx.shape)
        p.contourf(xx, yy, grid, levels=50, cmap='viridis', alpha=0.5)

    # Plot data points, cluster centers, and random point
    unique_labels = np.unique(y)
    palette = sns.color_palette("husl", len(unique_labels))  # Use a distinct color palette

    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['label'] = [f'Cluster {i}' for i in y]
    df = df.sort_values(by='label')
    
    sns.scatterplot(df, x='x1', y='x2', hue='label', palette=palette, ax=ax)
    sns.scatterplot(x=C[:, 0], y=C[:, 1], color='red', s=100, marker='o', ax=ax, label='Cluster Centers')
    sns.scatterplot(x=random_point[:, 0], y=random_point[:, 1], color='green', s=100, marker='o', ax=ax, label='Instance')


    if histories is not None:
        for h in histories:
            p.plot(h[:, 0], h[:, 1], color='black', linewidth=1, linestyle='dashed', markersize=2)
    if solutions is not None:
        for s in solutions:
            sns.scatterplot(x=s[:, 0], y=s[:, 1], color='purple', s=100, marker='o', ax=ax)

    plt.title(f'Heatmap of values for target cluster {target_cluster} ({fn.__name__})')
    # plt.axis('equal')
    if ax is None:
        plt.show()
