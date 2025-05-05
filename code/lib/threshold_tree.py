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
        self._dims = len(X[0])
        self._DTC_model = None
        self._DTC_tree = None
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

    def find_counterfactuals_random_forest(self, instance, target, threshold_change=0.1, robustness_factor=0.7, n_estimators=2, ratio_of_trees=0.5):
        """
        Find counterfactuals using Random Forest.

        Parameters
        ----------
        instance : Data point
        target : Target label
        threshold_change : Change from threshold when changing the feature
        robustness_factor : Factor to increase the robustness of the counterfactual
        n_estimators : Number of trees in the Random Forest
        ratio_of_trees : Ratio of trees to use for generating counterfactuals

        Returns
        -------
        List of counterfactuals
        """

        self._RF_instance = instance
        instance_label = self.model.predict([instance])[0]
        target_point = self.centers[target, :]
        
        clf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth=5, min_samples_leaf=1)
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

        inst = np.array([instance])
        targ = np.array([target_point])
        inst = inst.astype(np.float32)
        targ = targ.astype(np.float32)

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
                # print("mean: ", mean_threshold)
                cf[int(uniques[max_count_index][1])] = mean_threshold - threshold_change
                u_counts[max_count_index] = -1
            elif uniques[max_count_index][1] == 0:
                mean_threshold = np.mean(node_splits[np.all(node_splits[:,:2] == uniques[max_count_index], axis=1)][:,2])
                # print("mean: ", mean_threshold)
                cf[int(uniques[max_count_index][1])] = mean_threshold + threshold_change
                u_counts[max_count_index] = -1
            else:
                print("Could not finish counterfactual")
                break

        print("Counterfactual: ", cf)
        self._RF_cf = cf
        return self._RF_cf
    
    def plot_rf_tree(self):
        """
        Plot the Random Forest boundaries together with instance and counterfactuals.

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
        sns.scatterplot(x=[self._RF_cf[0]], y=[self._RF_cf[1]], color='blue', s=120, label='Counterfactual')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Random Forest Boundaries with Counterfactuals')
        plt.show()

    def find_counterfactuals_dtc(self, instance, target, threshold_change=0.1, robustness_factor=0.7, min_impurity_decrease=0.001):
        """
        Find counterfactuals using Decision Tree Classifier.

        Parameters
        ----------
        instance : Data point
        target : Target label
        threshold_change : Change from threshold when changing the feature
        robustness_factor : Factor to increase the robustness of the counterfactual
        min_impurity_decrease : Minimum impurity decrease to split a node using DTC

        Returns
        -------
        List of counterfactuals
        """

        instance_label = self.model.predict([instance])[0]
        target_point = self.centers[target, :]
        
        clf = DecisionTreeClassifier(random_state=42, min_impurity_decrease=min_impurity_decrease)
        clf.fit(self.X, self.y)
        print(f"DTC accuracy: {clf.score(self.X, self.y)}")

        n_leaves = clf.get_n_leaves()
        n_total_nodes = clf.tree_.node_count
        n_internal_nodes = n_total_nodes - n_leaves

        tree_model = clf.tree_
        self._DTC_model = clf
        self._DTC_tree = tree_model
        feature = tree_model.feature
        threshold = tree_model.threshold

        # Generate parent list for tree
        parent = np.full(n_total_nodes, -1, dtype=int)
        for i in range(n_total_nodes):
            if tree_model.children_left[i] != -1:
                parent[tree_model.children_left[i]] = i
            if tree_model.children_right[i] != -1:
                parent[tree_model.children_right[i]] = i

        # print("Parent list: ", parent)
        # print(f"Number of leaves: {n_leaves}")
        # print(f"Number of nodes: {n_total_nodes}")
        # print(f"Number of internal nodes: {n_internal_nodes}")

        # Find all leafs that are of the target class
        target_leafs = np.array([x for x in range(n_total_nodes) if tree_model.children_left[x] == -1 and np.argmax(tree_model.value[x]) == target])
        # print("Leaf Ids for target class: ", target_leafs)
        # print(f"Instance class: {instance_label}, point: {instance}")
        # print(f"Target center class: {target}, point: {target_point}")

        inst = np.array([instance])
        targ = np.array([target_point])
        inst = inst.astype(np.float32)
        targ = targ.astype(np.float32)
        inst_leaf_id = tree_model.apply(inst)

        cfs = np.zeros(shape=(target_leafs.shape[0], self._dims))
        cfs_prime = np.zeros(shape=(target_leafs.shape[0], self._dims))


        for j,l in enumerate(target_leafs):
            # print("Instance ID: ", inst_leaf_id)
            # print("Target node ID: ", l)
            # print("parent list: ", parent)

            target_node_indicator = np.zeros(shape=(n_total_nodes), dtype=int)
            curr_node = l
            # print("Leaf node: ", curr_node)
            while parent[curr_node] != -1:
                target_node_indicator[curr_node] = 1
                curr_node = parent[curr_node]
                # print("Parent node: ", curr_node)

            # inst = np.array([instance])
            # inst = inst.astype(np.float32)
            target_node_indicator[curr_node] = 1
            inst_node_indicator = np.array(tree_model.decision_path(inst).todense())[0]


            path_len = min(inst_node_indicator.shape[0], target_node_indicator.shape[0])
            path_equality = inst_node_indicator[:path_len] & target_node_indicator[:path_len]
            last_equal_parent = np.nonzero(path_equality)[0].max()
            # print("Index in tree of parent equality: ", last_equal_parent)

            temp = np.nonzero(target_node_indicator)[0]
            temp = temp[temp >= last_equal_parent]

            path_of_changes = set(temp)

            # print("-----------------------------------------------------------------------------------------")
            # print("Instance path: ", inst_node_indicator)
            # print("Target path: ", target_node_indicator)
            # print("Path of changes: ", path_of_changes)

            cf = instance.copy() # counterfactual

            curr_node = last_equal_parent
            i = 0
            while len(path_of_changes) > 1:
                # print("  CF:  ", cf)
                path_of_changes.remove(curr_node)
                if tree_model.children_left[curr_node] in path_of_changes:
                    # print(f"Change {i}: Left child")
                    if cf[feature[curr_node]] >= threshold[curr_node]:
                        cf[feature[curr_node]] = threshold[curr_node] - threshold_change
                    curr_node = tree_model.children_left[curr_node]
                elif tree_model.children_right[curr_node] in path_of_changes:
                    # print(f"Change {i}: Right child")
                    if cf[feature[curr_node]] < threshold[curr_node]:
                        cf[feature[curr_node]] = threshold[curr_node] + threshold_change
                    curr_node = tree_model.children_right[curr_node]
                else:
                    print("CHILD COULD NOT BE LOCATED!!!!!")
                    break
                i += 1

            # print("  CF:  ", cf)
            # print("")
            assert clf.predict([cf]) == target
            cf = np.array(cf)
            cf = cf.astype(np.float32)
            change = cf - instance
            change[np.isclose(change, 0, atol=0.00001)] = 0
            # print(f"Counterfactual change: {change}")
            # print(f"Counterfactual Tree index: {tree_model.apply(np.array([cf]))}")

            cfs_prime[j] = cf
            for i in range(self._dims):
                if change[i] != 0:
                    change_prime = instance[i] + change[i] + ((target_point[i] - cf[i]) * robustness_factor)
                    temp_cf = cfs_prime[j].copy()
                    temp_cf[i] = change_prime
                    # print(f"Pred: ", clf.predict([temp_cf2]))
                    if clf.predict([temp_cf]) == target:
                        cfs_prime[j][i] = change_prime
            # print("-----------------------------------------------------------------------------------------")
            cfs[j] = cf

        # print("Instance: ")
        # print(instance)
        # print("Counterfactuals: ")
        # print(cfs)
        # print("Counterfactuals': ")
        # print(cfs_prime)
        # print("")
        # all_cfs = np.concatenate((cfs, cfs_prime), axis=0)
        self._DTC_instance = instance
        self._DTC_cfs = cfs
        self._DTC_cfs_prime = cfs_prime
        return cfs, cfs_prime
    
    def print_dtc_tree(self):
        """
        Print the Decision Tree Classifier tree.

        Raises TypeError if the Decision Tree Classifier tree hasn't been trained.
        """
        if self._DTC_model is not None:
            tree.plot_tree(self._DTC_model, proportion=True, node_ids=True, impurity=False)
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
        
        if self._DTC_tree.feature[node] == 0:
            plt.plot([self._DTC_tree.threshold[node], self._DTC_tree.threshold[node]], [y_min, y_max], 'k-', lw=1)
            self._dtc_plot_decision_boundaries(self._DTC_tree.children_left[node], x_min, self._DTC_tree.threshold[node], y_min, y_max, depth + 1)
            self._dtc_plot_decision_boundaries(self._DTC_tree.children_right[node], self._DTC_tree.threshold[node], x_max, y_min, y_max, depth + 1)
        elif self._DTC_tree.feature[node] == 1:
            plt.plot([x_min, x_max], [self._DTC_tree.threshold[node], self._DTC_tree.threshold[node]], 'k-', lw=1)
            self._dtc_plot_decision_boundaries(self._DTC_tree.children_left[node], x_min, x_max, y_min, self._DTC_tree.threshold[node], depth + 1)
            self._dtc_plot_decision_boundaries(self._DTC_tree.children_right[node], x_min, x_max, self._DTC_tree.threshold[node], y_max, depth + 1)

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

    def find_counterfactuals_imm(self, instance, target, threshold_change=0.0001, robustness_factor=0.7):
        """
        Find counterfactuals using Decision Tree Classifier.

        Parameters
        ----------
        instance : Data point
        target : Target label
        threshold_change : Change from threshold when changing the feature
        robustness_factor : Factor to increase the robustness of the counterfactual 

        Returns
        -------
        List of counterfactuals
        """
        imm_model = imm()
        imm_model.fit(self.X, self.y, self.centers)
        self._IMM_model = imm_model
        target_point = self.centers[target, :]

        instance_path = imm_model.get_path(instance)
        target_path = imm_model.get_path(target_point)
        # print(f"Instance path: {instance_path}")
        # print(f"Target path: {target_path}\n")

        path_len = min(len(instance_path), len(target_path))
        path_equality = instance_path[:path_len] == target_path[:path_len]
        last_equal_parent = np.nonzero(path_equality)[0].max()
        # print("Index in tree of parent equality: ", last_equal_parent)
        path_of_changes = target_path[last_equal_parent:]
        # print("Path of changes: ", path_of_changes)

        cf = instance.copy()

        for i in range(len(path_of_changes) - 1):
            # print(f"Change {i}:")
            curr_node = path_of_changes[i]
            if curr_node.left == path_of_changes[i+1]:
                if cf[curr_node.feature] >= curr_node.threshold:
                    cf[curr_node.feature] = curr_node.threshold - threshold_change
            else:
                if cf[curr_node.feature] < curr_node.threshold:
                    cf[curr_node.feature] = curr_node.threshold + threshold_change

        change = cf - instance
        # print(change)
        change[np.isclose(change, 0, atol=0.00001)] = 0
        cf_prime = cf.copy()
        for i in range(self._dims):
            if change[i] != 0:
                change_prime = instance[i] + change[i] + ((target_point[i] - cf[i]) * robustness_factor)
                temp_cf = cf_prime.copy()
                temp_cf[i] = change_prime
                # print(f"Pred: ", clf.predict([temp_cf2]))
                if imm_model.predict(temp_cf) == target:
                    cf_prime[i] = change_prime
                else:
                    print("can't change")

        # print("Instance: ", instance)
        # print("Counterfactual: ", cf)
        # print("Counterfactual': ", cf_prime)
        # print("")
        # print("Original class: ", imm_model.predict(instance))
        # print("Counterfactual class: ", imm_model.predict(cf))
        self._IMM_instance = instance
        self._IMM_cf = cf
        self._IMM_cf_prime = cf_prime
        return np.array([cf]), np.array([cf_prime])

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
        sns.scatterplot(x=[self._IMM_cf_prime[0]], y=[self._IMM_cf_prime[1]], color='green', s=120, label='C\'')

        # Plot the decision boundaries
        self._imm_plot_decision_boundaries(tree, self.X[:, 0].min(), self.X[:, 0].max(), self.X[:, 1].min(), self.X[:, 1].max())
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('IMM Boundaries with Counterfactuals')

        plt.show()
        
