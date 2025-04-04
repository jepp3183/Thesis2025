from lib.imm import imm
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns



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

    def find_counterfactuals_dtc(self, instance, target, min_impurity_decrease=0.001, threshold_change=0.1, robustness_factor=0.7):
        """
        Find counterfactuals using Decision Tree Classifier.

        Parameters
        ----------
        instance : Data point
        target : Target label

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
            change[change < 0.00001] = 0
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
        print("")
        all_cfs = np.concatenate((cfs, cfs_prime), axis=0)
        self._DTC_instance = instance
        self._DTC_cfs = cfs
        self._DTC_cfs_prime = cfs_prime
        return all_cfs
    
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

    def plot_dtc_tree_colorized(self):
        """
        Plot the Decision Tree Classifier boundaries together with instance and counterfactuals.

        Raises TypeError if the Decision Tree Classifier tree hasn't been trained.
        """
        # Assuming X, y, instance, target_point, cf, and tree_model are already defined
        if self._DTC_model is None:
            raise TypeError("Decision Tree Classifier tree hasn't been trained.")
        elif self._DTC_instance.shape[0] != 2:
            raise ValueError("Only 2D data can be plotted.")

        # Plot the data points and clusters
        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(self.y)
        palette = sns.color_palette("husl", len(unique_labels))  # Use a distinct color palette
        sns.scatterplot(x=self.X[:, 0], y=self.X[:, 1], hue=self.y, palette=palette, alpha=0.6, edgecolor=None, legend='full')

        # Plot the initial point
        plt.scatter(self._DTC_instance[0], self._DTC_instance[1], color='red', s=100, label='Initial Point')

        # Plot the counterfactual
        for cf in self._DTC_cfs:
            plt.scatter(cf[0], cf[1], color='green', s=100, label='Counterfactual')

        for cf in self._DTC_cfs_prime:
            plt.scatter(cf[0], cf[1], color='yellow', s=100, label='C\'')

        # Plot the decision boundaries
        plot_step = 0.01
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
        Z = self._DTC_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

        # Add labels and legend
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title('Clusters, Decision Boundaries, Initial Point, and Counterfactual')
        plt.show()

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
        sns.scatterplot(x=self.X[:, 0], y=self.X[:, 1], hue=self.y, legend='full')
        sns.scatterplot(x=self._DTC_cfs[:,0], y=self._DTC_cfs[:,1], color='green', s=100, label='Counterfactual')
        sns.scatterplot(x=self._DTC_cfs_prime[:,0], y=self._DTC_cfs_prime[:,1], color='yellow', s=100, label='C\'')
        sns.scatterplot(x=[self._DTC_instance[0]], y=[self._DTC_instance[1]], color='red', s=100, label='Initial Point')

        # Plot the decision boundaries
        self._dtc_plot_decision_boundaries(root_node, self.X[:, 0].min(), self.X[:, 0].max(), self.X[:, 1].min(), self.X[:, 1].max())

        plt.show()

    def find_counterfactuals_imm(self, instance, target, threshold_change=0.0001):
        """
        Find counterfactuals using Decision Tree Classifier.

        Parameters
        ----------
        instance : Data point
        target : Target label

        Returns
        -------
        List of counterfactuals
        """
        imm_model = imm()
        imm_model.fit(self.X, self.y, self.centers)
        self._IMM_model = imm_model

        instance_path = imm_model.get_path(instance)
        target_path = imm_model.get_path(self.centers[target, :])
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

        # print("Instance: ", instance)
        # print("Counterfactual: ", cf)
        # print("")
        # print("Original class: ", imm_model.predict(instance))
        # print("Counterfactual class: ", imm_model.predict(cf))
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
        sns.scatterplot(x=self.X[:, 0], y=self.X[:, 1], hue=self.y, legend='full')
        sns.scatterplot(x=[self._IMM_cf[0]], y=[self._IMM_cf[1]], color='green', s=100, label='Counterfactual')
        sns.scatterplot(x=[self._IMM_instance[0]], y=[self._IMM_instance[1]], color='red', s=100, label='Initial Point')

        # Plot the decision boundaries
        self._imm_plot_decision_boundaries(tree, self.X[:, 0].min(), self.X[:, 0].max(), self.X[:, 1].min(), self.X[:, 1].max())

        plt.show()
        
        