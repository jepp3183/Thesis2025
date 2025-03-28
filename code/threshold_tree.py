from imm import imm
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class threshold_tree():

    def __init__(self, model, c, X, y):
        """
        Parameters
        ----------
        model : Trained model
        c : Cluster centers
        X : Training data
        y : Training labels
        """
        self.model = model
        self.centers = c
        self.X = X
        self.y = y
        self._dims = len(X[0])

    def find_counterfactuals_DTC(self, x, y, min_impurity_decrease=0.001, threshold_change=0.0000001, robustness_factor=0.7):
        """
        Find counterfactuals using Decision Tree Classifier.

        Parameters
        ----------
        x : Data point
        y : Target label

        Returns
        -------
        List of counterfactuals
        """
        
        clf = DecisionTreeClassifier(random_state=42, min_impurity_decrease=min_impurity_decrease)
        clf.fit(self.X, y)
        print(f"Accuracy: {clf.score(self.X, y)}")

        n_leaves = clf.get_n_leaves()
        n_total_nodes = clf.tree_.node_count
        n_internal_nodes = n_total_nodes - n_leaves

        tree_model = clf.tree_
        feature = tree_model.feature
        threshold = tree_model.threshold
        parent = np.full(n_total_nodes, -1, dtype=int)
        for i in range(n_total_nodes):
            if tree_model.children_left[i] != -1:
                parent[tree_model.children_left[i]] = i
            if tree_model.children_right[i] != -1:
                parent[tree_model.children_right[i]] = i

        # print("Parent list: ", parent)


        print(f"Number of leaves: {n_leaves}")
        print(f"Number of nodes: {n_total_nodes}")
        print(f"Number of internal nodes: {n_internal_nodes}")

        # Find all leafs that are of the target class
        target_leafs = np.array([x for x in range(n_total_nodes) if tree_model.children_left[x] == -1 and np.argmax(tree_model.value[x]) == target_class])
        print("Leaf Ids for target class: ", target_leafs)

        print(f"Instance class: {y[instance_index]}, point: {instance}")
        print(f"Target center class: {target_class}, point: {target_point}")

        inst = np.array([instance])
        targ = np.array([target_point])
        inst = inst.astype(np.float32)
        targ = targ.astype(np.float32)


        inst_node_indicator = np.array(tree_model.decision_path(inst).todense())[0]
        inst_leaf_id = tree_model.apply(inst)

        target_node_indicator = np.array(tree_model.decision_path(targ).todense())[0]
        target_leaf_id = tree_model.apply(targ)

        cfs = np.zeros(shape=(target_leafs.shape[0], dims))
        cfs_prime = np.zeros(shape=(target_leafs.shape[0], dims))


        for j,l in enumerate(target_leafs):
            print("Instance ID: ", inst_leaf_id)
            print("Target node ID: ", l)
            # print("parent list: ", parent)

            target_node_indicator = np.zeros(shape=(n_total_nodes), dtype=int)
            curr_node = l
            # print("Leaf node: ", curr_node)
            while parent[curr_node] != -1:
                target_node_indicator[curr_node] = 1
                curr_node = parent[curr_node]
                # print("Parent node: ", curr_node)

            target_node_indicator[curr_node] = 1


            inst = np.array([instance])
            inst = inst.astype(np.float32)


            inst_node_indicator = np.array(tree_model.decision_path(inst).todense())[0]


            path_len = min(inst_node_indicator.shape[0], target_node_indicator.shape[0])
            path_equality = inst_node_indicator[:path_len] & target_node_indicator[:path_len]
            last_equal_parent = np.nonzero(path_equality)[0].max()
            print("Index in tree of parent equality: ", last_equal_parent)

            temp = np.nonzero(target_node_indicator)[0]
            temp = temp[temp >= last_equal_parent]

            path_of_changes = set(temp)

            print("-----------------------------------------------------------------------------------------")
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
            print("")
            print(f"Counterfactual prediction: {clf.predict([cf])}")
            assert clf.predict([cf]) == target_class
            cf = np.array(cf)
            cf = cf.astype(np.float32)
            change = cf - instance
            change[change < 0.00001] = 0
            print(f"Counterfactual change: {change}")
            print(f"Counterfactual Tree index: {tree_model.apply(np.array([cf]))}")

            cfs_prime[j] = cf
            for i in range(dims):
                if change[i] != 0:
                    change_prime = instance[i] + change[i] + ((target_point[i] - cf[i]) * robustness_factor)
                    temp_cf = cfs_prime[j].copy()
                    temp_cf[i] = change_prime
                    # print(f"Pred: ", clf.predict([temp_cf2]))
                    if clf.predict([temp_cf]) == target_class:
                        cfs_prime[j][i] = change_prime
            print("-----------------------------------------------------------------------------------------")
            cfs[j] = cf

        print("Instance: ")
        print(instance)
        print("Counterfactuals: ")
        print(cfs)
        print("Counterfactuals': ")
        print(cfs_prime)
