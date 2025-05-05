import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def plot_cluster_cf(X, cluster_labels, cf, initial_point_idx):
    if X.shape[1] != 2:
        print("Data has more than 2 features. Using PCA!")
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        cf = pca.transform(cf)

    # Plot the clusters
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=cluster_labels, palette="deep", legend='full')

    # Plot the cluster centers
    sns.scatterplot(x=cf[:, 0], y=cf[:, 1], color='black', marker='X', s=100, label='Counterfactuals')

    # Highlight the initial point
    sns.scatterplot(x=[X[initial_point_idx, 0]], y=[X[initial_point_idx, 1]], color='red', marker='o', s=100, label='Initial Point')

    # Add legend and labels
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Cluster Plot with counterfactuals and Initial Point')
    plt.show()

def load_diabetes():
    path = "../data/diabetes.xls"
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def sparsity_fix(cf: np.ndarray, x: np.ndarray, model: KMeans) -> np.ndarray:
    org_cluster = model.predict(cf)
    fixed = cf.copy()
    features = list(range(x.shape[1]))

    print(f"Original counterfactual: {cf}")
    print(f"instance: {x}")
    print(f"Original cluster: {org_cluster}")

    while len(features) > 0:
        costs = np.array([get_reset_cost(fixed, cf, f) for f in features])
        feature_idx = np.argmin(costs)
        feature = features[feature_idx]

        value_backup = fixed[:, feature].copy()
        fixed[:, feature] = x[:, feature]

        print(f"costs: {costs}")
        print(f"Fixed: {fixed}")

        features.remove(feature)

        if model.predict(fixed) != org_cluster:
            fixed[:, feature] = value_backup
            print(f"break with {fixed}")
            break

    return fixed

def get_reset_cost(fixed: np.ndarray, cf: np.ndarray, feature) -> float:
    temp = fixed.copy()
    temp[:, feature] = cf[:, feature]
    return np.linalg.norm(temp - cf, axis=1).sum()