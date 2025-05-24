import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from random import sample


class TGCF_rf():

    def __init__(self, model, c, X, y):
        """
        Parameters
        ----------
        model : Trained clustering model
        c : Cluster centers
        X : Training data
        y : Training labels
        """
        self.model = model
        self.centers = c
        self.X = X
        self.y = y
        self.k = len(c)
        self._dims = len(X[0])
        self._RF_model = None
        self._RF_trees = None
        self._RF_instance = None
        self._RF_cf = None

    def fit(self, n_estimators=20, max_depth=None, min_samples_leaf=5):
        """
        Fit the Random Forest model to the data.

        Parameters
        ----------
        n_estimators : Number of trees in the Random Forest
        max_depth : Maximum depth of the trees
        min_samples_leaf : Minimum number of samples required to be at a leaf node
        """
        clf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        clf.fit(self.X, self.y)
        self._RF_model = clf
        print(f"Random Forest accuracy: {clf.score(self.X, self.y)}")

    def find_counterfactuals(self, instance, target, threshold_change=0.1, ratio_of_trees=1):
        """
        Find counterfactuals using Random Forest.

        Parameters
        ----------
        instance : Data point
        target : Target label
        threshold_change : Change from threshold when changing the feature
        n_estimators : Number of trees in the Random Forest
        ratio_of_trees : Ratio of trees to use for generating counterfactuals

        Returns
        -------
        List of counterfactuals
        """

        self._RF_instance = instance
        
        estimators = self._RF_model.estimators_
        _tree_models = [est.tree_ for est in estimators]

        amount_of_trees = math.ceil(len(_tree_models) * ratio_of_trees)

        chosen_trees = sample(_tree_models, amount_of_trees)

        parents = []
        target_leafs = []
        self._RF_trees = chosen_trees
        features = [tree_model.feature for tree_model in chosen_trees]
        thresholds = [tree_model.threshold for tree_model in chosen_trees]
        n_total_nodes = [tree_model.node_count for tree_model in chosen_trees]

        for j, (t_tree_model, t_n_total_nodes) in enumerate(zip(chosen_trees,n_total_nodes)):
            temp_parent = np.full(t_n_total_nodes, -1, dtype=int)
            for i in range(t_n_total_nodes):
                if t_tree_model.children_left[i] != -1:
                    temp_parent[t_tree_model.children_left[i]] = i
                if t_tree_model.children_right[i] != -1:
                    temp_parent[t_tree_model.children_right[i]] = i
            parents.append(temp_parent)
            target_leafs.append(np.array([x for x in range(t_n_total_nodes) if t_tree_model.children_left[x] == -1 and np.argmax(t_tree_model.value[x]) == target]))

        node_splits = np.empty(shape=(0, 3)) # For every threshold, we have the feature, threshold value and the path direction
        
        for i, (t_tree_model, t_parent, t_target_leafs, t_features, t_threshold, t_n_total_nodes) in enumerate(zip(chosen_trees, parents, target_leafs, features, thresholds, n_total_nodes)):
            for j, l in enumerate(t_target_leafs):
                target_node_indicator = np.zeros(shape=(t_n_total_nodes), dtype=int)
                curr_node = l
                while t_parent[curr_node] != -1:
                    target_node_indicator[curr_node] = 1
                    curr_node = t_parent[curr_node]
                target_node_indicator[curr_node] = 1

                path = set(np.nonzero(target_node_indicator)[0])
                curr_node = 0
                while len(path) > 1:
                    path.remove(curr_node)
                    if t_tree_model.children_left[curr_node] in path:
                        node_splits = np.append(node_splits, [[1,t_features[curr_node],t_threshold[curr_node]]], axis=0)
                        curr_node = t_tree_model.children_left[curr_node]
                    elif t_tree_model.children_right[curr_node] in path:
                        node_splits = np.append(node_splits, [[0,t_features[curr_node],t_threshold[curr_node]]], axis=0)
                        curr_node = t_tree_model.children_right[curr_node]
                    elif path == {}:
                        break
                    else :
                        print("CHILD COULD NOT BE LOCATED!!!!!")

        uniques, u_counts = np.unique(node_splits[:,:2], axis=0, return_counts=True)

        cf = instance.copy() # counterfactual

        while self._RF_model.predict([cf]) != target:
            max_count_index = np.argmax(u_counts)
            if u_counts[max_count_index] == -1:
                break
            elif uniques[max_count_index][1] == 1:
                mean_threshold = np.mean(node_splits[np.all(node_splits[:,:2] == uniques[max_count_index], axis=1)][:,2])
                cf[int(uniques[max_count_index][1])] = mean_threshold - threshold_change
                u_counts[max_count_index] = -1
            elif uniques[max_count_index][1] == 0:
                mean_threshold = np.mean(node_splits[np.all(node_splits[:,:2] == uniques[max_count_index], axis=1)][:,2])
                cf[int(uniques[max_count_index][1])] = mean_threshold + threshold_change
                u_counts[max_count_index] = -1
            else:
                break

        self._RF_cf = cf.reshape(1, -1)
        return self._RF_cf
    
    def print_tree(self):
        """
        Print the TF trees.

        Raises TypeError if the RF trees haven't been trained.
        """
        print("Not yet implemented for Random Forests.")

    def plot_tree(self):
        """
        Plot the Random Forest results with instance and counterfactuals.

        Raises TypeError if the Random Forest tree hasn't been trained.
        """
        if self._RF_model is None:
            raise TypeError("Random Forest tree hasn't been trained.")
        elif self._RF_instance.shape[0] != 2:
            raise ValueError("Only 2D data can be plotted.")

        # Plot the data points
        df = pd.DataFrame(self.X, columns=['x1', 'x2'])
        df['label'] = [f'Cluster {i}' for i in self.y]
        df = df.sort_values(by='label')

        sns.scatterplot(df, x='x1', y='x2', hue='label', palette="Set2", legend='full')
        # Plot centroids
        sns.scatterplot(x=self.centers[:, 0], y=self.centers[:, 1], color='black', s=70, label='Cluster Centers')
        sns.scatterplot(x=[self._RF_instance[0]], y=[self._RF_instance[1]], color='red', s=120, label='Instance')
        sns.scatterplot(x=self._RF_cf[:, 0], y=self._RF_cf[:, 1], color='blue', s=120, label='Counterfactual')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Random Forest Boundaries with Counterfactuals')
        plt.show()

    def plausibility_fix(self, cfs, instance, target_cluster, method, plausibility_factor = 0.7):
        cfs_prime = np.zeros(shape=(cfs.shape[0], self._dims))
        target_point = self.centers[target_cluster, :]

        if method == "DTC":
            predict_fn = lambda x: self.dtc_predict([x])[0]
        elif method == "IMM" and self._IMM_model is not None:
            predict_fn = lambda x: self._IMM_model.predict(x)
        else:
            raise ValueError("Invalid method. Use 'DTC' or 'IMM'.")

        for j, cf in enumerate(cfs):
            change = cf - instance
            change[np.isclose(change, 0, atol=0.00001)] = 0

            cfs_prime[j] = cf
            for i in range(self._dims):
                if change[i] != 0:
                    change_prime = instance[i] + change[i] + ((target_point[i] - cf[i]) * plausibility_factor)
                    temp_cf = cfs_prime[j].copy()
                    temp_cf[i] = change_prime
                    if predict_fn(temp_cf) == target_cluster:
                        cfs_prime[j][i] = change_prime

        if method == "IMM":
            self._IMM_cf_prime = cfs_prime
        elif method == "DTC":
            self._DTC_cfs_prime = cfs_prime
        return cfs_prime
