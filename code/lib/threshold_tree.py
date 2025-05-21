from lib.imm import imm
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
import pandas as pd
import math
from random import sample
from collections import Counter


class ThresholdTree():

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
        self._thresholds = None
        self._features = None
        self._children_left = None
        self._children_right = None
        self._parents = None
        self._values = None
        self._n_node_samples = None
        self._DTC_model = None
        self._DTC_instance = None
        self._DTC_cfs = None
        self._DTC_cfs_prime = None
        self._IMM_model = None
        self._IMM_instance = None
        self._IMM_cf = None
        self._IMM_cf_prime = None
        self._RF_model = None
        self._RF_trees = None
        self._RF_instance = None
        self._RF_cf = None

    def fit_dtc(self, improve_tree_fidelity, min_impurity_decrease=0.001):
        clf = DecisionTreeClassifier(random_state=42, min_impurity_decrease=min_impurity_decrease)#, min_samples_leaf=self.X.shape[0] // 20)
        clf.fit(self.X, self.y)
        print(f"DTC accuracy: {clf.score(self.X, self.y)}")

        self._DTC_model = clf
        self._thresholds = clf.tree_.threshold.copy()
        self._features = clf.tree_.feature.copy()
        self._children_left = clf.tree_.children_left.copy()
        self._children_right = clf.tree_.children_right.copy()
        self._values = clf.tree_.value.copy()
        self._n_node_samples = clf.tree_.n_node_samples.copy()

        # Generate parent list for tree
        self._parents = np.full(self._thresholds.shape[0], -1, dtype=int)
        for i in range(self._thresholds.shape[0]):
            if self._children_left[i] != -1:
                self._parents[self._children_left[i]] = i
            if self._children_right[i] != -1:
                self._parents[self._children_right[i]] = i
        
        sample_leaf_indices = clf.apply(self.X)
        leaf_indices = np.array([x for x in range(self._thresholds.shape[0]) if self._children_left[x] == -1])
        leaf_sample_lists = [[i for i in range(sample_leaf_indices.shape[0]) if sample_leaf_indices[i] == leaf_indices[j]] for j in range(clf.get_n_leaves())]
    
        if improve_tree_fidelity:
            for i,j in enumerate(leaf_indices):
                self.update_tree(j,leaf_sample_lists[i])

    def fit_imm(self):
        imm_model = imm()
        imm_model.fit(self.X, self.y, self.centers)
        self._IMM_model = imm_model


    def find_counterfactuals_rf(self, instance, target, threshold_change=0.1, n_estimators=20, ratio_of_trees=1):
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
        
        clf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth=None, min_samples_leaf=5)
        clf.fit(self.X, self.y)
        print(f"Random Forest accuracy: {clf.score(self.X, self.y)}")

        estimators = clf.estimators_
        _tree_models = [est.tree_ for est in estimators]

        amount_of_trees = math.ceil(len(_tree_models) * ratio_of_trees)

        chosen_trees = sample(_tree_models, amount_of_trees)

        parents = []
        target_leafs = []
        self._RF_model = clf
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

        while clf.predict([cf]) != target:
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
    
    def plot_rf_tree(self):
        """
        Plot the Random Forest results with instance and counterfactuals.

        Raises TypeError if the Random Forest tree hasn't been trained.
        """
        if self._RF_model is None:
            raise TypeError("Random Forest tree hasn't been trained.")
        elif self._RF_instance.shape[0] != 2:
            raise ValueError("Only 2D data can be plotted.")

        # Assuming imm_model is an instance of the imm class and has been fitted
        root_node = 0

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
        plt.title('Random Forest Boundaries with Counterfactuals')
        plt.show()

    def update_tree(self, leaf_index, leaf_sample_list, cut_span_threshold = 0.66):
        # Create path list for leaf index
        target_node_indicator = np.zeros(shape=(self._thresholds.shape[0]), dtype=int)
        curr_node = leaf_index
        while self._parents[curr_node] != -1:
            target_node_indicator[curr_node] = 1
            curr_node = self._parents[curr_node]
        target_node_indicator[curr_node] = 1
        path = set(np.nonzero(target_node_indicator)[0])

        node_splits = np.empty(shape=(0, 3)) # List of threshold splits on path with evaluation value
        curr_node = 0
        while len(path) > 1:
            path.remove(curr_node)
            if self._children_left[curr_node] in path:
                node_splits = np.append(node_splits, [[1,self._features[curr_node],self._thresholds[curr_node]]], axis=0)
                curr_node = self._children_left[curr_node]
            elif self._children_right[curr_node] in path:
                node_splits = np.append(node_splits, [[0,self._features[curr_node],self._thresholds[curr_node]]], axis=0)
                curr_node = self._children_right[curr_node]
            else:
                print("CHILD COULD NOT BE LOCATED!!!!!")
                break



        n_dims = self.X.shape[1]
        left_splits = []
        right_splits = []
        for i in range(n_dims):
            list_left = node_splits[np.all(node_splits[:,:2] == np.array([[1,i]]), axis=1)][:,2]
            list_right = node_splits[np.all(node_splits[:,:2] == np.array([[0,i]]), axis=1)][:,2]
            left_splits.append(
                np.min(list_left) if len(list_left) > 0 else None
            )
            right_splits.append(
                np.max(list_right) if len(list_right) > 0 else None
            )

        min_dimension = np.min(self.X, axis=0)
        max_dimension = np.max(self.X, axis=0)    

        dim_ranges = list(zip(
            [md if ls is None else ls for md,ls in zip(max_dimension, left_splits)], 
            [md if rs is None else rs for md,rs in zip(min_dimension, right_splits)]
        ))

        leaf_data = np.take(self.X, leaf_sample_list, axis=0)
        dim_cut_properties = [
            self.calc_dimension_cut_properties(dim_ranges[d], leaf_data[:, d]) 
            for d in range(n_dims)
        ]

        dim_to_cut = -1
        best_score = 1
        for i, res in enumerate(dim_cut_properties):
            if res == None: continue # None means bad cut
            if res[0] < best_score and res[0] <= cut_span_threshold:
                dim_to_cut = i
                best_score = res[0]
        
        if dim_to_cut == -1: return

        
        _, space_left, space_right, cut_left, cut_right = dim_cut_properties[dim_to_cut]
        threshold = cut_left if space_left >= space_right else cut_right
        
        
        self._thresholds = np.append(self._thresholds,[-1])
        self._thresholds = np.append(self._thresholds,[-1])
        self._thresholds[leaf_index] = threshold
        
        self._features = np.append(self._features,[-1])
        self._features = np.append(self._features,[-1])
        self._features[leaf_index] = dim_to_cut
        
        self._children_left = np.append(self._children_left,[-1])
        self._children_left = np.append(self._children_left,[-1])
        left_leaf_index = len(self._thresholds) - 2
        self._children_left[leaf_index] = left_leaf_index
        
        self._children_right = np.append(self._children_right,[-1])
        self._children_right = np.append(self._children_right,[-1])
        right_leaf_index = len(self._thresholds) - 2 + 1
        self._children_right[leaf_index] = right_leaf_index
        
        self._parents = np.append(self._parents,[leaf_index])
        self._parents = np.append(self._parents,[leaf_index])

        label_left = []
        label_right = []
        
        sample_left = []
        sample_right = []

        for sample_index in leaf_sample_list:
            label = self.y[sample_index]
            if self.X[sample_index][dim_to_cut] <= threshold:
                label_left.append(label)
                sample_left.append(sample_index)
            else:
                label_right.append(label)
                sample_right.append(sample_index)

        self._n_node_samples = np.append(self._n_node_samples,[len(sample_left)])
        self._n_node_samples = np.append(self._n_node_samples,[len(sample_right)])
        
        def get_percentage(labels):
            total = len(labels)
            amount = Counter(labels)

            return np.array([[[ (amount.get(i, 0) / total) for i in range(self.k) ]]])
        
        self._values = np.append(self._values, get_percentage(label_left), axis=0) 
        self._values = np.append(self._values, get_percentage(label_right), axis=0)
        
        self.update_tree(left_leaf_index, sample_left)
        self.update_tree(right_leaf_index, sample_right)

    def dtc_predict(self, data_points):
        
        def recursive_predict(node, data_point):
            if self._children_left[node] == -1:
                value = self._values[node][0]
                return np.argmax(value)
            
            threshold = self._thresholds[node]
            feature = self._features[node]

            if data_point[feature] <= threshold:
                return recursive_predict(self._children_left[node], data_point)
            else:
                return recursive_predict(self._children_right[node], data_point)
        
        results = []
        for point in data_points:
            results.append(recursive_predict(0,point))

        return results
        
    def calc_dimension_cut_properties(self, dim_range_tuple, leaf_data, threshold = 0.95):
        """
        For the given dimension range and leaf data, returns the span score, space left, space right, cut left and cut right.
        The span score is the ratio of the data range to the dimension range.
        The space left and right are the distances from the data range to the dimension range.
        The cut left and right are the cuts that can be made.

        Parameters
        ----------
        dim_range_tuple : Tuple of min and max values of the dimension range
        leaf_data : Leaf data
        threshold : Threshold for the span score

        Returns
        -------
        span score, space left, space right, cut left and cut right
        """
        _, counts = np.unique(self.y, return_counts=True)
        smallest_cluster_size = np.min(counts)
        if len(leaf_data) <= smallest_cluster_size // 2: return None

        dim_min, dim_max = dim_range_tuple
        dim_range = abs(dim_max - dim_min)

        low_idx = 0
        high_idx = len(leaf_data) - 1
        while high_idx - low_idx + 1 > len(leaf_data) * threshold:
            low_idx += 1
            high_idx -= 1

        leaf_data_sorted = np.sort(leaf_data)
        data_high = leaf_data_sorted[high_idx]
        data_low = leaf_data_sorted[low_idx]
        data_range = abs(data_high - data_low)

        space_left = data_low - dim_min
        space_right = dim_max - data_high

        cut_left = (leaf_data_sorted[low_idx] + leaf_data_sorted[low_idx - 1]) / 2
        cut_right = (leaf_data_sorted[high_idx] + leaf_data_sorted[high_idx + 1]) / 2

        span_score = data_range / dim_range

        return span_score, space_left, space_right, cut_left, cut_right
    
    def calc_node_indicator(self, data_point):  
        """
        Calculate the node indicator for the given data point.

        Parameters
        ----------
        data_point : Data point

        Returns
        -------
        Node indicator - 0/1 encoding for visited nodes on path to leaf
        """
        node_indicator = np.zeros(shape=(self._thresholds.shape[0]), dtype=int)
        curr_node = 0
        node_indicator[curr_node] = 1
        while self._children_left[curr_node] != -1:
            if data_point[self._features[curr_node]] <= self._thresholds[curr_node]:
                curr_node = self._children_left[curr_node]
            elif data_point[self._features[curr_node]] > self._thresholds[curr_node]:
                curr_node = self._children_right[curr_node]
            node_indicator[curr_node] = 1
        return node_indicator 

    def find_leaf_centers_dtc(self, target_leafs):
        """
        Find the centers of the target leafs.

        Parameters
        ----------
        target_leafs : Target leafs to find centers for

        Returns
        -------
        Centers of the target leafs
        """
        centers = np.zeros(shape=(target_leafs.shape[0], self._dims))

        for k, l in enumerate(target_leafs):
            target_node_indicator = np.zeros(shape=(self._thresholds.shape[0]), dtype=int)
            curr_node = l
            while self._parents[curr_node] != -1:
                target_node_indicator[curr_node] = 1
                curr_node = self._parents[curr_node]
            target_node_indicator[curr_node] = 1
            path = set(np.nonzero(target_node_indicator)[0])

            node_splits = np.empty(shape=(0, 3)) # List of threshold splits on path with evaluation value
            curr_node = 0
            while len(path) > 1:
                path.remove(curr_node)
                if self._children_left[curr_node] in path:
                    node_splits = np.append(node_splits, [[1,self._features[curr_node],self._thresholds[curr_node]]], axis=0)
                    curr_node = self._children_left[curr_node]
                elif self._children_right[curr_node] in path:
                    node_splits = np.append(node_splits, [[0,self._features[curr_node],self._thresholds[curr_node]]], axis=0)
                    curr_node = self._children_right[curr_node]
                else:
                    print("CHILD COULD NOT BE LOCATED!!!!!")
                    break

            n_dims = self.X.shape[1]
            left_splits = []
            right_splits = []
            for i in range(n_dims):
                list_left = node_splits[np.all(node_splits[:,:2] == np.array([[1,i]]), axis=1)][:,2]
                list_right = node_splits[np.all(node_splits[:,:2] == np.array([[0,i]]), axis=1)][:,2]
                left_splits.append(
                    np.min(list_left) if len(list_left) > 0 else None
                )
                right_splits.append(
                    np.max(list_right) if len(list_right) > 0 else None
                )

            min_dimension = np.min(self.X, axis=0)
            max_dimension = np.max(self.X, axis=0)    

            dim_ranges = list(zip(
                [md if ls is None else ls for md,ls in zip(max_dimension, left_splits)], 
                [md if rs is None else rs for md,rs in zip(min_dimension, right_splits)]
            ))
            center = np.array([(left + right) / 2 for left, right in dim_ranges])
            centers[k] = center

        return centers

    def find_counterfactuals_dtc(
        self,
        instance,
        target,
        threshold_change=0.1,
        filter_target_leafs=True
    ):
        """
        Find counterfactuals using Decision Tree Classifier.

        Parameters
        ----------
        instance : Data point
        target : Target label
        threshold_change : Change from threshold when changing the feature
        filter_target_leafs : Whether to filter target leafs based on the model prediction of the leaf centers

        Returns
        -------
        List of counterfactuals
        """
        assert self._DTC_model is not None, "DTC model has not been trained yet."

        # Find all leafs that are of the target class 
        target_leafs = np.array([x for x in range(self._thresholds.shape[0]) if self._children_left[x] == -1 and np.argmax(self._values[x]) == target])

        # Filter found target leafs based on clustering membership of the leaf centers
        if filter_target_leafs:
            leaf_centers = self.find_leaf_centers_dtc(target_leafs)
            target_leafs = target_leafs[self.model.predict(leaf_centers) == target]
        
        cfs = np.zeros(shape=(target_leafs.shape[0], self._dims))


        for j,l in enumerate(target_leafs):

            target_node_indicator = np.zeros(shape=(self._thresholds.shape[0]), dtype=int)
            curr_node = l
            while self._parents[curr_node] != -1:
                target_node_indicator[curr_node] = 1
                curr_node = self._parents[curr_node]

            target_node_indicator[curr_node] = 1
            inst_node_indicator = self.calc_node_indicator(instance)


            path_len = min(inst_node_indicator.shape[0], target_node_indicator.shape[0])
            path_equality = inst_node_indicator[:path_len] & target_node_indicator[:path_len]
            last_equal_parent = np.nonzero(path_equality)[0].max()

            temp = np.nonzero(target_node_indicator)[0]
            temp = temp[temp >= last_equal_parent]

            path_of_changes = set(temp)

            cf = instance.copy() # counterfactual

            curr_node = last_equal_parent
            i = 0
            while len(path_of_changes) > 1:
                path_of_changes.remove(curr_node)
                if self._children_left[curr_node] in path_of_changes:
                    if cf[self._features[curr_node]] >= self._thresholds[curr_node]:
                        cf[self._features[curr_node]] = self._thresholds[curr_node] - threshold_change
                    curr_node = self._children_left[curr_node]
                elif self._children_right[curr_node] in path_of_changes:
                    if cf[self._features[curr_node]] < self._thresholds[curr_node]:
                        cf[self._features[curr_node]] = self._thresholds[curr_node] + threshold_change
                    curr_node = self._children_right[curr_node]
                else:
                    print("CHILD COULD NOT BE LOCATED!!!!!")
                    break
                i += 1

            cf = np.array(cf)
            cf = cf.astype(np.float32)
            cfs[j] = cf

        self._DTC_instance = instance
        self._DTC_cfs = cfs
        return cfs 

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
    
    def print_dtc_tree(self):
        """
        Print the Decision Tree Classifier tree.

        Raises TypeError if the Decision Tree Classifier tree hasn't been trained.
        """
        if self._DTC_model is not None:
            tree.plot_tree(self._DTC_model, proportion=True, node_ids=True, impurity=True)
            # plt.savefig('fig1.png', dpi = 3000) # Save tree for inspection
            plt.show()
        else:
            raise TypeError("Decision Tree Classifier tree hasn't been trained.")

    def _dtc_plot_decision_boundaries(self, node, x_min, x_max, y_min, y_max, depth=0):
        """
        Private method for plotting decision boundaries of the DTC tree.
        """
        if node == -1:
            return
        
        if self._features[node] == 0:
            plt.plot([self._thresholds[node], self._thresholds[node]], [y_min, y_max], 'k-', lw=1)
            self._dtc_plot_decision_boundaries(self._children_left[node], x_min, self._thresholds[node], y_min, y_max, depth + 1)
            self._dtc_plot_decision_boundaries(self._children_right[node], self._thresholds[node], x_max, y_min, y_max, depth + 1)
        elif self._features[node] == 1:
            plt.plot([x_min, x_max], [self._thresholds[node], self._thresholds[node]], 'k-', lw=1)
            self._dtc_plot_decision_boundaries(self._children_left[node], x_min, x_max, y_min, self._thresholds[node], depth + 1)
            self._dtc_plot_decision_boundaries(self._children_right[node], x_min, x_max, self._thresholds[node], y_max, depth + 1)

    def plot_dtc_tree(self):
        """
        Plot the DTC boundaries together with instance and counterfactuals.

        Raises TypeError if the DTC tree hasn't been trained.
        """
        if self._DTC_model is None:
            raise TypeError("DTC tree hasn't been trained.")
        elif self._DTC_instance.shape[0] != 2:
            raise ValueError("Only 2D data can be plotted.")

        # Assuming imm_model is an instance of the imm class and has been fitted
        root_node = 0

        # Plot the data points
        df = pd.DataFrame(self.X, columns=['x1', 'x2'])
        df['label'] = [f'Cluster {i}' for i in self.y]
        df = df.sort_values(by='label')

        sns.scatterplot(df, x='x1', y='x2', hue='label', palette="Set2", legend='full')
        # Plot centroids
        sns.scatterplot(x=self.centers[:, 0], y=self.centers[:, 1], color='black', s=70, label='Cluster Centers')
        sns.scatterplot(x=[self._DTC_instance[0]], y=[self._DTC_instance[1]], color='red', s=120, label='Instance')
        sns.scatterplot(x=self._DTC_cfs[:,0], y=self._DTC_cfs[:,1], color='blue', s=120, label='Counterfactual')
        sns.scatterplot(x=self._DTC_cfs_prime[:,0], y=self._DTC_cfs_prime[:,1], color='green', s=120, label='C\'')

        # Plot the decision boundaries
        self._dtc_plot_decision_boundaries(root_node, self.X[:, 0].min(), self.X[:, 0].max(), self.X[:, 1].min(), self.X[:, 1].max())
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Tree Classifier Boundaries with Counterfactuals')
        plt.show()

    def find_counterfactuals_imm(self, instance, target, threshold_change=0.0001):
        """
        Find counterfactuals using Decision Tree Classifier.

        Parameters
        ----------
        instance : Data point
        target : Target label
        threshold_change : Change from threshold when changing the feature

        Returns
        -------
        List of counterfactuals
        """
        assert self._IMM_model is not None, "IMM model has not been trained yet."
        
        target_point = self.centers[target, :]

        instance_path = self._IMM_model.get_path(instance)
        target_path = self._IMM_model.get_path(target_point)

        path_len = min(len(instance_path), len(target_path))
        path_equality = instance_path[:path_len] == target_path[:path_len]
        last_equal_parent = np.nonzero(path_equality)[0].max()
        path_of_changes = target_path[last_equal_parent:]

        cf = instance.copy()

        for i in range(len(path_of_changes) - 1):
            curr_node = path_of_changes[i]
            if curr_node.left == path_of_changes[i+1]:
                if cf[curr_node.feature] >= curr_node.threshold:
                    cf[curr_node.feature] = curr_node.threshold - threshold_change
            else:
                if cf[curr_node.feature] < curr_node.threshold:
                    cf[curr_node.feature] = curr_node.threshold + threshold_change


        self._IMM_instance = instance
        self._IMM_cf = cf
        return np.array([cf])

    def print_imm_tree(self):
        """
        Print the IMM tree.

        Raises TypeError if the IMM tree hasn't been trained.
        """
        if self._IMM_model is not None:
            print(self._IMM_model.write_tree())
        else:
            raise TypeError("IMM tree hasn't been trained.")
        
    def _imm_plot_decision_boundaries(self, node, x_min, x_max, y_min, y_max, depth=0):
        """
        Private method for plotting decision boundaries of the IMM tree.
        """
        if node is None or node.cluster is not None:
            return
        
        if node.feature == 0:
            plt.plot([node.threshold, node.threshold], [y_min, y_max], 'k-', lw=1)
            self._imm_plot_decision_boundaries(node.left, x_min, node.threshold, y_min, y_max, depth + 1)
            self._imm_plot_decision_boundaries(node.right, node.threshold, x_max, y_min, y_max, depth + 1)
        elif node.feature == 1:
            plt.plot([x_min, x_max], [node.threshold, node.threshold], 'k-', lw=1)
            self._imm_plot_decision_boundaries(node.left, x_min, x_max, y_min, node.threshold, depth + 1)
            self._imm_plot_decision_boundaries(node.right, x_min, x_max, node.threshold, y_max, depth + 1)

    def plot_imm_tree(self):
        """
        Plot the IMM boundaries together with instance and counterfactuals.

        Raises TypeError if the IMM tree hasn't been trained.
        """
        if self._IMM_model is None:
            raise TypeError("IMM tree hasn't been trained.")
        elif self._IMM_instance.shape[0] != 2:
            raise ValueError("Only 2D data can be plotted.")

        # Assuming imm_model is an instance of the imm class and has been fitted
        tree = self._IMM_model.tree

        # Plot the data points
        df = pd.DataFrame(self.X, columns=['x1', 'x2'])
        df['label'] = [f'Cluster {i}' for i in self.y]
        df = df.sort_values(by='label')

        sns.scatterplot(df, x='x1', y='x2', hue='label', palette="Set2", legend='full')
        # Plot centroids
        sns.scatterplot(x=self.centers[:, 0], y=self.centers[:, 1], color='black', s=70, label='Cluster Centers')
        sns.scatterplot(x=[self._IMM_instance[0]], y=[self._IMM_instance[1]], color='red', s=120, label='Instance')
        sns.scatterplot(x=[self._IMM_cf[0]], y=[self._IMM_cf[1]], color='blue', s=120, label='Counterfactual')
        sns.scatterplot(x=[self._IMM_cf_prime[0][0]], y=[self._IMM_cf_prime[0][1]], color='green', s=120, label='C\'')

        # Plot the decision boundaries
        self._imm_plot_decision_boundaries(tree, self.X[:, 0].min(), self.X[:, 0].max(), self.X[:, 1].min(), self.X[:, 1].max())
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('IMM Boundaries with Counterfactuals')

        plt.show()
        
