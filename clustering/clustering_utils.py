"""
clustering_utils.py

Utility functions for the customer segmentation clustering notebook.
This file assumes that the preprocessing notebook already exported a scaled dataset:
    ../datasets/info_clustering_ready.csv

The clustering notebook should use these functions to:
- load the dataset
- prepare X for clustering
- test KMeans for different k values
- run the final KMeans model
- visualize clusters with PCA and UMAP
- summarize and export results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA


def load_clustering_data(filepath="../datasets/info_clustering_ready.csv"):
    """
    Load the preprocessed and scaled dataset for clustering.

    Parameters
    ----------
    filepath : str
        Path to the clustering-ready CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded clustering dataframe.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()


def prepare_clustering_matrix(df, id_col="customer_id"):
    """
    Separate customer_id from clustering features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_col : str
        Customer identifier column.

    Returns
    -------
    customer_ids : pd.Series or None
        Customer IDs if available.
    X : pd.DataFrame
        Feature matrix for clustering.
    """
    if id_col in df.columns:
        customer_ids = df[id_col].copy()
        X = df.drop(columns=[id_col])
    else:
        customer_ids = None
        X = df.copy()

    print(f"Clustering matrix prepared: {X.shape[0]} rows, {X.shape[1]} features")
    return customer_ids, X


def evaluate_kmeans_range(X, min_k=2, max_k=10, random_state=42):
    """
    Evaluate KMeans for a range of k values using inertia and silhouette score.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Clustering features.
    min_k : int
        Minimum number of clusters.
    max_k : int
        Maximum number of clusters.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with k, inertia and silhouette score.
    """
    results = []

    print(f"Computing metrics for k from {min_k} to {max_k}")
    print("-" * 50)

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=50,
            max_iter=500
        )

        labels = model.fit_predict(X)
        inertia = model.inertia_
        silhouette = silhouette_score(X, labels)

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette": silhouette
        })

        print(f"k={k}: Inertia={inertia:.0f}, Silhouette={silhouette:.4f}")

    return pd.DataFrame(results)


def plot_elbow_silhouette(results_df):
    """
    Plot inertia and silhouette score for each k.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from evaluate_kmeans_range().
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(results_df["k"], results_df["inertia"], marker="o")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method and Silhouette Score")

    ax2 = ax1.twinx()
    ax2.plot(results_df["k"], results_df["silhouette"], marker="x", linestyle="--")
    ax2.set_ylabel("Silhouette Score")

    plt.grid(True)
    plt.show()


def run_final_kmeans(df, X, final_k=3, id_col="customer_id", random_state=42):
    """
    Run final KMeans model and add cluster labels to the original dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Original clustering dataframe with customer_id.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    final_k : int
        Final number of clusters.
    id_col : str
        ID column name.
    random_state : int
        Random seed.

    Returns
    -------
    model : KMeans
        Fitted KMeans model.
    df_clustered : pd.DataFrame
        DataFrame with kmeans_cluster column.
    score : float
        Final silhouette score.
    """
    model = KMeans(
        n_clusters=final_k,
        random_state=random_state,
        n_init=50,
        max_iter=500
    )

    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    df_clustered = df.copy()
    df_clustered["kmeans_cluster"] = labels

    print(f"Final k: {final_k}")
    print(f"Final Silhouette Score: {score:.4f}")

    return model, df_clustered, score


def cluster_size_summary(df_clustered, cluster_col="kmeans_cluster"):
    """
    Create a summary table with count and percentage per cluster.

    Parameters
    ----------
    df_clustered : pd.DataFrame
        Clustered dataframe.
    cluster_col : str
        Cluster label column.

    Returns
    -------
    pd.DataFrame
        Cluster size summary.
    """
    counts = df_clustered[cluster_col].value_counts().sort_index()
    percentages = df_clustered[cluster_col].value_counts(normalize=True).sort_index() * 100

    summary = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2)
    })

    return summary


def get_centroids(model, X, cluster_col="kmeans_cluster"):
    """
    Return KMeans centroids as a dataframe.

    Parameters
    ----------
    model : KMeans
        Fitted KMeans model.
    X : pd.DataFrame
        Feature matrix used for clustering.
    cluster_col : str
        Name for the cluster index.

    Returns
    -------
    pd.DataFrame
        Centroids dataframe.
    """
    centroids = pd.DataFrame(
        model.cluster_centers_,
        columns=X.columns
    )

    centroids[cluster_col] = range(model.n_clusters)
    centroids = centroids.set_index(cluster_col)

    return centroids


def plot_pca_clusters(X, labels, title="PCA - KMeans Clusters", random_state=42):
    """
    Plot clusters in 2D using PCA.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    labels : array-like
        Cluster labels.
    title : str
        Plot title.
    random_state : int
        Random seed.
    """
    pca = PCA(n_components=2, random_state=random_state)
    embedding = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))

    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            label=f"Cluster {cluster}"
        )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Explained variance by PC1 and PC2: {pca.explained_variance_ratio_.sum():.4f}")


def plot_umap_clusters(X, labels, title="UMAP - KMeans Clusters", random_state=42):
    """
    Plot clusters in 2D using UMAP.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    labels : array-like
        Cluster labels.
    title : str
        Plot title.
    random_state : int
        Random seed.
    """
    try:
        import umap.umap_ as umap
    except ImportError:
        print("UMAP is not installed. Install it with: pip install umap-learn")
        return

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        random_state=random_state
    )

    embedding = reducer.fit_transform(X)

    plt.figure(figsize=(10, 7))

    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            label=f"Cluster {cluster}"
        )

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def silhouette_plot(X, labels, title="Silhouette Plot"):
    """
    Plot silhouette coefficients for each cluster.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    labels : array-like
        Cluster labels.
    title : str
        Plot title.
    """
    silhouette_avg = silhouette_score(X, labels)
    sample_values = silhouette_samples(X, labels)

    unique_labels = sorted(np.unique(labels))

    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10

    for cluster in unique_labels:
        cluster_values = sample_values[labels == cluster]
        cluster_values.sort()

        size_cluster = cluster_values.shape[0]
        y_upper = y_lower + size_cluster

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_values,
            alpha=0.7,
            label=f"Cluster {cluster}"
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster))
        y_lower = y_upper + 10

    ax.axvline(
        x=silhouette_avg,
        linestyle="--",
        label=f"Average = {silhouette_avg:.4f}"
    )

    ax.set_title(title)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.legend()
    ax.grid(True)
    plt.show()


def save_clustered_data(df_clustered, filepath="../datasets/info_clustering_kmeans.csv"):
    """
    Save clustered dataframe to CSV.

    Parameters
    ----------
    df_clustered : pd.DataFrame
        Dataframe with cluster labels.
    filepath : str
        Output path.
    """
    output_dir = os.path.dirname(filepath)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_clustered.to_csv(filepath, index=False)
    print(f"File saved: {filepath}")
