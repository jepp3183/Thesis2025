import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

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