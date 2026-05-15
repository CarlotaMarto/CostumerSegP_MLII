from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, fcluster
import matplotlib.pyplot as plt
import pandas as pd


def read_newdata(filepath="info_clustering.csv"):
    """
    Reads customer data from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'customer_info.csv' in the current directory.

    Returns
    -------
    pd.DataFrame
        The loaded customer information DataFrame.
    """
    try:
        customer_info = pd.read_csv(filepath)
        return customer_info
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()


def read_outliers(filepath="outlier_dataset.csv"):
    """
    Reads customer data from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'customer_info.csv' in the current directory.

    Returns
    -------
    pd.DataFrame
        The loaded customer information DataFrame.
    """
    try:
        customer_info = pd.read_csv(filepath)
        return customer_info
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()



def get_color_map(labels, cmap_name='tab10'):
    unique_labels = sorted(np.unique(labels))
    color_map = {label: cm.get_cmap(cmap_name)(i % 10) for i, label in enumerate(unique_labels)}
    return [color_map[label] for label in labels]


def run_kmeans(df, k, exclude_cols=None, random_state=42):
    """
    Runs KMeans clustering on the input DataFrame, excluding specified columns.

    Returns:
        labels: cluster assignments
        centroids: cluster centers
        score: silhouette score
    """
    if exclude_cols is None:
        exclude_cols = []

    features = df.drop(columns=exclude_cols).values
    model = KMeans(n_clusters=k, random_state=random_state)
    labels = model.fit_predict(features)
    centroids = model.cluster_centers_
    score = silhouette_score(features, labels)
    return labels, centroids, score


def elbow_and_silhouette(df, max_k=10, exclude_cols=None, random_state=42):
    """
    Plots Elbow and Silhouette score curves for a range of k values.
    
    Returns:
        inertias: list of inertia values
        silhouettes: list of silhouette scores
    """
    if exclude_cols is None:
        exclude_cols = []

    data = df.drop(columns=exclude_cols).values
    inertias = []
    silhouettes = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(data, labels)
        silhouettes.append(score)
        print(f"K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={score:.4f}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia", color="tab:blue")
    ax1.plot(range(2, max_k + 1), inertias, marker='o', color="tab:blue", label="Inertia")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:green")
    ax2.plot(range(2, max_k + 1), silhouettes, marker='x', linestyle='--', color="tab:green", label="Silhouette")
    ax2.tick_params(axis='y', labelcolor="tab:green")

    plt.title("Elbow and Silhouette Scores")
    fig.tight_layout()
    plt.show()

    return inertias, silhouettes




def visualize_umap(embedding, labels=None, title='UMAP Projection', xlabel='Component 1', ylabel='Component 2', cmap_name='tab10'):
    plt.figure(figsize=(10, 8))

    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10)
    else:
        unique_labels = sorted(np.unique(labels))
        colors = get_color_map(labels, cmap_name)

        for label in unique_labels:
            idx = labels == label
            color = colors[np.where(labels == label)[0][0]]
            plt.scatter(embedding[idx, 0], embedding[idx, 1], c=[color], s=10, label=f"Cluster {label}")

        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_pca(embedding, labels=None, title='PCA Projection', xlabel='PC 1', ylabel='PC 2', cmap_name='tab10'):
    plt.figure(figsize=(10, 8))

    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10)
    else:
        unique_labels = sorted(np.unique(labels))
        colors = get_color_map(labels, cmap_name)

        for label in unique_labels:
            idx = labels == label
            color = colors[np.where(labels == label)[0][0]]
            plt.scatter(embedding[idx, 0], embedding[idx, 1], c=[color], s=10, label=f"Cluster {label}")

        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def silhouette_plot(X, cluster_labels, title='Silhouette Plot', cmap_name='tab10'):
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    unique_labels = sorted(np.unique(cluster_labels))
    colors = get_color_map(cluster_labels, cmap_name)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10

    for label in unique_labels:
        ith_values = sample_silhouette_values[cluster_labels == label]
        ith_values.sort()
        size_i = ith_values.shape[0]
        y_upper = y_lower + size_i
        color = colors[np.where(cluster_labels == label)[0][0]]

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_values,
                         facecolor=color, edgecolor=color, alpha=0.7, label=f"Cluster {label}")

        ax.text(-0.05, y_lower + 0.5 * size_i, str(label))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--", label=f"Avg = {silhouette_avg:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_dendrogram(linked, title="Hierarchical Clustering on K-Means Centroids"):
    plt.figure(figsize=(10, 6))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title(title)
    plt.show()

def assign_macro_clusters(linked, num_clusters):
    """
    Assigns macro-cluster labels based on a dendrogram cut.

    Returns:
        array of macro-cluster labels
    """
    return fcluster(linked, t=num_clusters, criterion='maxclust')



