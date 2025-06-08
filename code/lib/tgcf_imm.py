from lib.imm import imm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class TGCF_imm():

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
        self._IMM_model = None
        self._IMM_instance = None
        self._IMM_cf = None
        self._IMM_cf_prime = None

    def fit(self):
        imm_model = imm()
        imm_model.fit(self.X, self.y, self.centers)
        self._IMM_model = imm_model
    
    def plausibility_fix(self, cfs, instance, target_cluster, plausibility_factor = 0.7):
        cfs_prime = np.zeros(shape=(cfs.shape[0], self._dims))
        target_point = self.centers[target_cluster, :]

        for j, cf in enumerate(cfs):
            change = cf - instance
            change[np.isclose(change, 0, atol=0.00001)] = 0

            cfs_prime[j] = cf
            for i in range(self._dims):
                if change[i] != 0:
                    change_prime = instance[i] + change[i] + ((target_point[i] - cf[i]) * plausibility_factor)
                    temp_cf = cfs_prime[j].copy()
                    temp_cf[i] = change_prime
                    if self._IMM_model.predict(temp_cf) == target_cluster:
                        cfs_prime[j][i] = change_prime

        self._IMM_cf_prime = cfs_prime
        return cfs_prime

    def find_counterfactuals(self, instance, target, threshold_change=0.0001):
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

        cf = np.array([cf])

        self._IMM_instance = instance
        self._IMM_cf = cf
        return cf
    
    def _imm_accuracy(self):
        """
        Calculate the accuracy of the IMM model subject to cluster labels.

        Parameters
        ----------

        Returns
        -------
        Accuracy of the IMM model
        """

        if self._IMM_model is None:
            raise TypeError("IMM model hasn't been trained yet.")
        
        predictions = np.array([self._IMM_model.predict(x) for x in self.X])
        accuracy = np.sum(predictions == self.y) / len(self.y)
        return accuracy

    def print_tree(self):
        """
        Print the IMM tree.

        Raises TypeError if the IMM tree hasn't been trained.
        """
        if self._IMM_model is not None:
            print(self._IMM_model.write_tree())
        else:
            raise TypeError("IMM tree hasn't been trained.")
        
    def _plot_decision_boundaries(self, node, x_min, x_max, y_min, y_max, depth=0):
        """
        Private method for plotting decision boundaries of the IMM tree.
        """
        if node is None or node.cluster is not None:
            return
        
        if node.feature == 0:
            plt.plot([node.threshold, node.threshold], [y_min, y_max], 'k-', lw=1)
            self._plot_decision_boundaries(node.left, x_min, node.threshold, y_min, y_max, depth + 1)
            self._plot_decision_boundaries(node.right, node.threshold, x_max, y_min, y_max, depth + 1)
        elif node.feature == 1:
            plt.plot([x_min, x_max], [node.threshold, node.threshold], 'k-', lw=1)
            self._plot_decision_boundaries(node.left, x_min, x_max, y_min, node.threshold, depth + 1)
            self._plot_decision_boundaries(node.right, x_min, x_max, node.threshold, y_max, depth + 1)

    def plot_tree(self):
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
        sns.scatterplot(x=[self._IMM_cf[0][0]], y=[self._IMM_cf[0][1]], color='blue', s=120, label='Counterfactual')
        sns.scatterplot(x=[self._IMM_cf_prime[0][0]], y=[self._IMM_cf_prime[0][1]], color='green', s=120, label='C\'')

        # Plot the decision boundaries
        self._plot_decision_boundaries(tree, self.X[:, 0].min(), self.X[:, 0].max(), self.X[:, 1].min(), self.X[:, 1].max())
        plt.xlabel('x1')
        plt.ylabel('x2')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.legend().set_visible(False)
        plt.title(f'IMM splits with counterfactuals (accuracy: {(self._imm_accuracy()):.3f})')

        plt.show()
        
